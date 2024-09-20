from pathlib import Path
import subprocess
import os
import sys
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

def save_data(expt,run = "*"):

    basepath = Path("/home/149/ab8992/topographic-NIWs/rundirs") / expt.split("_")[0] / expt / "archive" / f"output{run}"
    print("basepath ",basepath)
    u = xr.open_mfdataset(str(basepath) + "/u.nc", decode_times = False,decode_cf = False).sel(yh = slice(-250,250)).sel(xq = 250,method = "nearest").u.load()
    ufar = xr.open_mfdataset(str(basepath) + "/u.nc", decode_times = False,decode_cf = False).sel(yh = slice(-250,250)).isel(xq = 0).u.load()
    v = xr.open_mfdataset(str(basepath) + "/v.nc", decode_times = False,decode_cf = False).sel(xh = slice(-250,250)).sel(yq = 250,method = "nearest").v.load()

    eEastFar = xr.open_mfdataset(str(basepath) + "/e.nc", decode_times = False,decode_cf = False).sel(yh = slice(-250,250)).isel(xh = 0).e.load()

    eEast = xr.open_mfdataset(str(basepath) + "/e.nc", decode_times = False,decode_cf = False).sel(yh = slice(-250,250)).sel(xh = 250,method = "nearest").e.load()
    eNorth = xr.open_mfdataset(str(basepath) + "/e.nc", decode_times = False,decode_cf = False).sel(xh = slice(-250,250)).sel(yh = 250,method = "nearest").e.load()

    pEast = al.calculate_pressure(eEast,u.zl)
    pEastFar = al.calculate_pressure(eEastFar,u.zl)
    pNorth = al.calculate_pressure(eNorth,u.zl)

    pEast = pEast - pEast.isel(time = 0)
    pEastFar = pEastFar - pEastFar.isel(time = 0)
    pNorth = pNorth - pNorth.isel(time = 0)

    out = xr.merge([u - ufar,v,(pEast - pEastFar).rename("pEast"),pNorth.rename("pNorth")]).load()
    outpath = Path("/g/data/v45/ab8992/bottom-niws-outputs/sept2024-bniw-outputs/") / expt.split("_")[0] / expt
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    if run == "*":
        run = "all"
    out.to_netcdf(outpath / f"{expt}_{run}.nc")
    return 

def postprocess(rundir):
    # Rundir is a path object
    basepath = rundir.parent.parent.parent.absolute() ## Go to root of the repo
    archive = basepath / "outputdir" / rundir.name
    print(archive)
    print(archive.absolute())
    (archive / "zonal").mkdir(exist_ok=True,parents = True)
    (archive / "merid").mkdir(exist_ok=True,parents = True)

    ## Find forcing duration
    duration = 10
    file = open(rundir / "MOM_override")
    for line in file.readlines():
        if "FORCING_DURATION" in line:
            duration = int(line.split(" ")[-1])
            break
    file.close()
    ## Check whether we've already run for long enough

    outputs = list(rundir.glob("archive/output*"))

   
    ## Now cut out the data we want to keep and save to archive

    current_output = rundir / "archive" / f"output{(len(outputs) - 1):03}" 
    print(current_output)

    save_data(rundir.name,run = f"{(len(outputs) - 1):03}" )
    # for file in ["u.nc","v.nc","e.nc","taux.nc"]:
    #     data = xr.open_dataarray(current_output / file)
    #     if "xq" in data.dims:
    #         zonal = data.sel(yh = slice(-10,10))
    #         merid = data.isel(xq = slice(0,10))
    #     elif "yq" in data.dims:
    #         zonal = data.sel(yq = slice(-10,10))
    #         merid = data.isel(xh = slice(0,10))
    #     else:
    #         zonal = data.sel(yh = slice(-10,10))
    #         merid = data.isel(xh = slice(0,10))

    #     zonal.to_netcdf(archive / "zonal" / f"{file.split('.nc')[0]}_{len(outputs) - 1}.nc")
    #     merid.to_netcdf(archive / "merid" / f"{file.split('.nc')[0]}_{len(outputs) - 1}.nc")


    ## Now run the model for the rest of the duration
    if len(outputs) == 1:

        with open(rundir / "MOM_override", 'a') as file:
            file.write("#override WIND_CONFIG = 'const'\n")
            file.write("#override CONST_WIND_TAUX = 0\n")
            file.write("#override CONST_WIND_TAUY = 0\n")
        overwrite_in_file(str((rundir / "input.nml").absolute()),
                          "    hours = ",
                          "    hours = "
                          )
        subprocess.run(
            f"payu run -f -n 10",shell= True,cwd = str(rundir)
        )


parser = argparse.ArgumentParser()
parser.add_argument('-e', '--expt', type=str, help='Experiment',default = "common")
args = parser.parse_args()
if __name__ == "__main__":
    client = Client(threads_per_worker = 2)

    postprocess(Path(args.expt))

