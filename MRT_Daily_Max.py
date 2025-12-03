"""
MRT Daily Maximum Calculator
Calculates Mean Radiant Temperature for urban/rural/mixed environments.

Uses tasmax and tas (daily mean) - estimates tasmin as: tasmin = 2*tas - tasmax

Usage:
    python MRT_Daily_Max.py MODEL SCENARIO [PERIOD]

Examples:
    python MRT_Daily_Max.py CESM2 Historical
    python MRT_Daily_Max.py CESM2 ssp585 NearFuture
"""

import numpy as np
import pandas as pd
import sys
import os

try:
    import xarray as xr
    HAS_XR = True
except ImportError:
    HAS_XR = False

SBC = 5.67e-8

ENV = {
    "urban": (0.12, 0.97),
    "rural": (0.25, 0.92),
    "mixed": (0.18, 0.95)
}


def calc_rad(tmax, tmin, hurs, lat, doy):
    tas = (tmax + tmin) / 2
    tas_k = tas + 273.15
    satvap = 0.6108 * np.exp((17.27 * tas) / (tas + 237.3))
    avp = (hurs / 100) * satvap
    emiss_atm = 1.08 * (1 - np.exp(-avp ** (tas_k / 2016)))
    lw = SBC * tas_k**4 * emiss_atm

    lat_r = np.radians(lat)
    dec = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(-np.tan(lat_r) * np.tan(dec))
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    ra = (24 * 60 / np.pi) * 0.0820 * dr * (
        ws * np.sin(lat_r) * np.sin(dec) +
        np.cos(lat_r) * np.cos(dec) * np.sin(ws)
    )
    
    # Ensure dtr is non-negative (minimum 0.1 to avoid issues)
    dtr = np.maximum(tmax - tmin, 0.1)
    transmiss = 0.7 * (1 - np.exp(-0.004 * dtr ** 2.4))
    sw = np.maximum(ra * 1e6 / 86400 * transmiss, 0)

    return lw, sw


def calc_mrt(tmax, tmin, hurs, lat, doy, alb, emis):
    lw_sky, sw = calc_rad(tmax, tmin, hurs, lat, doy)
    lw_ground = emis * SBC * (tmax + 273.15)**4
    sw_absorbed = (1 - alb) * sw
    r_tot = lw_sky + lw_ground + sw_absorbed
    return (r_tot / (SBC * 2)) ** 0.25 - 273.15


def process_nc(model, scenario, period=None):
    if not HAS_XR:
        raise ImportError("xarray required")

    ts = {
        "Full": range(4014, 31390),
        "NearFuture": range(4015, 11315),
        "MidFuture": range(15330, 26645),
        "FarFuture": range(26645, 31390)
    }

    bp = "/work/nsco/breinsvold/cmip6"

    def get_path(var):
        return bp + "/" + scenario + "/" + var + "/Downscaled_CMIP6_" + var + "_Nebraska_" + scenario + "_" + model + ".nc"

    print("Loading " + model + " / " + scenario + "...")
    print("  TASMAX: " + get_path("tasmax"))
    print("  TAS:    " + get_path("tas"))
    print("  HURS:   " + get_path("hurs"))

    tmax_ds = xr.open_dataset(get_path("tasmax"))
    tas_ds = xr.open_dataset(get_path("tas"))
    hurs_ds = xr.open_dataset(get_path("hurs"))

    if period and period in ts:
        tmax = tmax_ds["tasmax"].isel(time=ts[period])
        tas = tas_ds["tas"].isel(time=ts[period])
        hurs = hurs_ds["hurs"].isel(time=ts[period])
        print("  Period: " + period)
    else:
        tmax = tmax_ds["tasmax"]
        tas = tas_ds["tas"]
        hurs = hurs_ds["hurs"]

    # Estimate tasmin from tas and tasmax
    # tas = (tasmax + tasmin) / 2  =>  tasmin = 2*tas - tasmax
    # Ensure tasmin doesn't exceed tasmax
    tmin = np.minimum(2 * tas - tmax, tmax - 0.1)
    print("  Estimating TASMIN from TAS and TASMAX")
    print("  Shape: " + str(tmax.shape))

    lat2d = xr.broadcast(tmax.lat, tmax)[0]
    doy = xr.DataArray(
        pd.to_datetime(tmax.time.values).dayofyear,
        dims="time",
        coords={"time": tmax.time}
    )
    doy3d = doy.broadcast_like(tmax)

    print("Calculating MRT (vectorized)...")

    mrt_results = {}
    for env, (alb, emis) in ENV.items():
        mrt_results["MRT_" + env.capitalize()] = xr.apply_ufunc(
            calc_mrt, tmax, tmin, hurs, lat2d, doy3d, alb, emis, dask="allowed"
        )
    mrt_results["Urban_Rural_Diff"] = mrt_results["MRT_Urban"] - mrt_results["MRT_Rural"]

    print("Converting to DataFrame...")

    records = []
    times = tmax.time.values
    lats = tmax.lat.values
    lons = tmax.lon.values

    for ti, t in enumerate(times):
        for li, lat in enumerate(lats):
            for lo, lon in enumerate(lons):
                record = {
                    "time": pd.Timestamp(t),
                    "lat": lat,
                    "lon": lon,
                    "MRT_Urban": float(mrt_results["MRT_Urban"].isel(time=ti, lat=li, lon=lo)),
                    "MRT_Rural": float(mrt_results["MRT_Rural"].isel(time=ti, lat=li, lon=lo)),
                    "MRT_Mixed": float(mrt_results["MRT_Mixed"].isel(time=ti, lat=li, lon=lo)),
                    "Urban_Rural_Diff": float(mrt_results["Urban_Rural_Diff"].isel(time=ti, lat=li, lon=lo))
                }
                records.append(record)

        if (ti + 1) % 100 == 0:
            print("  Processed " + str(ti + 1) + "/" + str(len(times)) + " timesteps...")

    df = pd.DataFrame(records)

    out_dir = bp + "/" + scenario + "/mrt"
    os.makedirs(out_dir, exist_ok=True)

    out_path = out_dir + "/Downscaled_CMIP6_mrt_Nebraska_" + scenario + "_" + model + ".csv"
    df.to_csv(out_path, index=False, float_format="%.2f")
    print("Saved " + out_path + " (" + str(len(df)) + " records)")

    return df


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python MRT_Daily_Max.py MODEL SCENARIO [PERIOD]")
        print("")
        print("Examples:")
        print("  python MRT_Daily_Max.py CESM2 Historical")
        print("  python MRT_Daily_Max.py CESM2 ssp585 NearFuture")
        print("")
        print("PERIOD options: Full, NearFuture, MidFuture, FarFuture")
        sys.exit(1)

    model = sys.argv[1]
    scenario = sys.argv[2]
    period = sys.argv[3] if len(sys.argv) > 3 else None

    print("==================================================")
    print("  MRT Daily Maximum Calculator")
    print("==================================================")
    print("  Model:    " + model)
    print("  Scenario: " + scenario)
    if period:
        print("  Period:   " + period)
    print("")

    process_nc(model, scenario, period)
