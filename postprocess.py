from pathlib import Path
import subprocess
import os
import sys
import xarray as xr
import argparse
from dask.distributed import Client

basepath = Path.cwd().absolute()


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


    for file in ["u_10min.nc","v_10min.nc","e_10min.nc","taux_10min.nc","pbo_10min.nc"]:
        data = xr.open_dataarray(current_output / file)
        if "xq" in data.dims:
            zonal = data.sel(yh = slice(-10,10))
            merid = data.isel(xq = slice(0,10))
        elif "yq" in data.dims:
            zonal = data.sel(yq = slice(-10,10))
            merid = data.isel(xh = slice(0,10))
        else:
            zonal = data.sel(yh = slice(-10,10))
            merid = data.isel(xh = slice(0,10))

        zonal.to_netcdf(archive / "zonal" / f"{file.split('.nc')[0]}_{len(outputs) - 1}.nc")
        merid.to_netcdf(archive / "merid" / f"{file.split('.nc')[0]}_{len(outputs) - 1}.nc")


    with open(rundir / "MOM_override", 'a') as file:
        file.write("#override WIND_CONFIG = 'const'\n")
        file.write("#override CONST_WIND_TAUX = 0\n")
        file.write("#override CONST_WIND_TAUY = 0\n")

    ## Now run the model for the rest of the duration
    subprocess.run(
        f"payu run -f -n 5",shell= True,cwd = str(rundir)
    )

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--expt', type=str, help='Experiment',default = "common")
args = parser.parse_args()
if __name__ == "__main__":
    client = Client(threads_per_worker = 2)

    postprocess(Path(args.expt))

