from pathlib import Path
import os
import numpy as np
import xarray as xr
import argparse
from dask.distributed import Client
import autolib as al
basepath = Path.cwd().absolute()

def overwrite_in_file(filepath,old,new):
    file = open(filepath,"r")
    lines = file.readlines()
    file.close()
    for i in range(len(lines)):
        if old in lines[i]:
            lines[i] = new + "\n"
    file = open(filepath,"w")
    file.writelines(lines)
    file.close()
    return

def save_all_data(expt,run = "*"):
    """
    Same as in postprocess, but this time save all in one file for each expt. Also do East as well as west!!
    """
    if "forcing_width" not in expt:
        expt_base_name = expt.split("_")[0]
    else:
        expt_base_name = "forcing_width"
    levels = np.append(np.array([50,150]),np.full(19,200))

    basepath = Path("/home/149/ab8992/topographic-NIWs/rundirs") / expt_base_name / expt / "archive" / f"output{run}"
    print("basepath ",basepath)
    u = xr.open_mfdataset(str(basepath) + "/u.nc", decode_times = False,decode_cf = False).sel(yh = slice(-250,250)).sel(xq = [-250,250],method = "nearest").u.load()
    ufar = xr.open_mfdataset(str(basepath) + "/u.nc", decode_times = False,decode_cf = False).sel(yh = slice(-250,250)).isel(xq = 0).u.load()
    v = xr.open_mfdataset(str(basepath) + "/v.nc", decode_times = False,decode_cf = False).sel(xh = slice(-250,250)).sel(yq = [-250,250],method = "nearest").v.load()

    eEastFar = xr.open_mfdataset(str(basepath) + "/e.nc", decode_times = False,decode_cf = False).sel(yh = slice(-250,250)).isel(xh = 0).e.load()

    eEast = xr.open_mfdataset(str(basepath) + "/e.nc", decode_times = False,decode_cf = False).sel(yh = slice(-250,250)).sel(xh = 250,method = "nearest").e.load()
    eNorth = xr.open_mfdataset(str(basepath) + "/e.nc", decode_times = False,decode_cf = False).sel(xh = slice(-250,250)).sel(yh = 250,method = "nearest").e.load()
    eWest = xr.open_mfdataset(str(basepath) + "/e.nc", decode_times = False,decode_cf = False).sel(yh = slice(-250,250)).sel(xh = -250,method = "nearest").e.load()
    eSouth = xr.open_mfdataset(str(basepath) + "/e.nc", decode_times = False,decode_cf = False).sel(xh = slice(-250,250)).sel(yh = -250,method = "nearest").e.load()
    
    pEast = al.calculate_pressure(eEast,u.zl)
    pWest = al.calculate_pressure(eWest,u.zl)
    pEastFar = al.calculate_pressure(eEastFar,u.zl)
    pNorth = al.calculate_pressure(eNorth,u.zl)
    pSouth = al.calculate_pressure(eSouth,u.zl)

    pEast = pEast - pEast.isel(time = 0)
    pWest = pWest - pWest.isel(time = 0)
    pEastFar = pEastFar - pEastFar.isel(time = 0)
    pNorth = pNorth - pNorth.isel(time = 0)
    pSouth = pSouth - pSouth.isel(time = 0)

    out = xr.merge([u - ufar,
                    v,
                    (pEast - pEastFar).rename("pEast"),
                    pSouth.rename("pSouth"),
                    pNorth.rename("pNorth"),
                    (pWest - pEastFar).rename("pWest"),
                    ]).load()
    
    out.assign_coords(zl = levels)

    # Now integrate `out` to get the energy fluxes
    endtime = int(u.time[-1].values)

    ridge_width = 12500
    forcing_width = 100000
    if "forcing_width" in expt:
        forcing_width = float(expt.split("_")[-1])
    if "width" in expt:
        ridge_width = float(expt.split("_")[-1])


    EF_EW = 1000 * (
    (out.u.isel(xq = 1) * out.pEast - out.u.isel(xq = 0) * out.pWest) ## Sum the East and West energy fluxes
    .integrate("time")
    .integrate("yh")
    .integrate("zl")
    /  (endtime * 500000 * ridge_width)
    )

    EF_NS = 1000 * (
    (out.v.isel(yq = 1) * out.pNorth - out.v.isel(yq = 0) * out.pSouth) ## Sum the East and West energy fluxes
    .integrate("time")
    .integrate("xh")
    .integrate("zl")
    /  (endtime * 500000 * forcing_width)
    ) 

    outpath = Path("/g/data/v45/ab8992/postprocessed_bottomniws/") / expt_base_name / expt
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    if run == "*":
        run = "all"
    xr.merge(
        [out,EF_EW.rename("EF_EW"),EF_NS.rename("EF_NS")]
        ).to_netcdf(outpath / f"{expt}_{run}.nc",mode = "w")
    print(f"Success {expt} {run}")
    return 


parser = argparse.ArgumentParser()
parser.add_argument('-e', '--expt', type=str, help='Experiment',default = "common")
args = parser.parse_args()
if __name__ == "__main__":
    client = Client(threads_per_worker = 2)
    basepath = Path("/home/149/ab8992/topographic-NIWs/rundirs")
    for i in basepath.glob(args.expt +  "/*"):
        if args.expt in i.name:
            print(i.name)
            save_all_data(i.name)

