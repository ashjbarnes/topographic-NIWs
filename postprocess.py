from pathlib import Path
import os
import sys
import xarray as xr
basepath = Path.cwd().absolute()


def postprocess(rundir):
    # Rundir is a path object
    archive = basepath / "outputdir" / rundir.name
    (archive / "zonal").mkdir(exist_ok=True)
    (archive / "merid").mkdir(exist_ok=True)

    ## Find forcing duration

    file = open(rundir / "MOM_override")
    for line in file.readlines():
        if "forcing_duration" in line:
            duration = int(line.split(" ")[-1])
            break
    file.close()
    ## Check whether we've already run for long enough

    outputs = list(rundir.glob("archive/output*"))
    if len(outputs) == int(duration):
        ## Need to switch off forcing in the MOM_override file
        with open(rundir / "MOM_override", 'a') as file:
            file.write("#override WIND_CONFIG = 'const'\n")
            file.write("#override CONST_WIND_TAUX = 0\n")
            file.write("#override CONST_WIND_TAUY = 0\n")
    
    ## Now cut out the data we want to keep and save to archive

    current_output = rundir / "archive" / f"output{(len(outputs) - 1):03}" 
    print(current_output)


    for file in ["u_10min.nc","v_10min.nc","eta_10min.nc","taux_10min.nc"]:
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