from matplotlib import rc
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import os
import xarray as xr
import subprocess
import matplotlib.pyplot as plt
import shutil

def setup_mom6(name,tname,overrides = [],walltime = None,common_forcing = False,default_dir = "default"):
    ## If common forcing is provided, set another input folder that contains the windstress for all runs in this experiment
    default_dir = f"/home/149/ab8992/bottom_near_inertial_waves/automated/{default_dir}/*"
    
    if name in os.listdir("/home/149/ab8992/bottom_near_inertial_waves/automated"):
        shutil.rmtree("/home/149/ab8992/bottom_near_inertial_waves/automated/" + name)
    
    # subprocess.run(["/home/149/ab8992/tools/myscripts/automated_mom6/copydir",name])
    subprocess.run(f"cp {default_dir} /home/149/ab8992/bottom_near_inertial_waves/automated/{name} -r",shell = True)

    ## We've created the new directory with name 'name'
    # Now need to edit config file to point to the name of the topography configuraation 'tname'
    file = open("/home/149/ab8992/bottom_near_inertial_waves/automated/" + name + "/config.yaml","r")
    config = file.readlines()
    file.close()
    ##Iterate through and find jobname and input
    
    for line in range(len(config)):
        if "jobname" in config[line][0:10]:
            config[line] = "jobname: " + name  + "\n"
            
        if "input" in config[line][0:10]:
            if common_forcing == False:
                config[line] = "input: /g/data/v45/ab8992/mom6_channel_configs/" + tname + "\n"
            else:
                config[line] = f"input:\n     - /g/data/v45/ab8992/mom6_channel_configs/{tname}\n     - /g/data/v45/ab8992/mom6_channel_configs/{common_forcing}\n"
            
        if "walltime" in config[line][0:10] and walltime != None:
            config[line] = "walltime: " + str(walltime)
    file = open("/home/149/ab8992/bottom_near_inertial_waves/automated/" + name + "/config.yaml","w")
    file.writelines(config)
    
    
    ## Update override file
    
    file = open("/home/149/ab8992/bottom_near_inertial_waves/automated/" + name + "/MOM_override","r")
    override_file = file.readlines()
    file.close()
    
    for i in overrides:
        override_file.append("#override " + i + str("\n"))
        
    file = open("/home/149/ab8992/bottom_near_inertial_waves/automated/" + name + "/MOM_override","w")
    file.writelines(override_file)
    
    return

def save_topo(x,y,STRESS_X,STRESS_Y,layerdensities,eta,tname,savewind =True,savedensities = True):
   
    outfolder = "/g/data/v45/ab8992/mom6_channel_configs/" + tname
    
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
        
        
    else:
        if "driver-layers-topo-densities.nc" in os.listdir(outfolder):
            os.remove(outfolder + "/driver-layers-topo-densities.nc")
        if "storm_windstress.nc" in os.listdir(outfolder):
            os.remove(outfolder + "/storm_windstress.nc")
    
    if savedensities == True:
        driverfilename = outfolder+'/driver-layers-topo-densities.nc'
        ncOutput = Dataset(driverfilename, 'w', format='NETCDF4')
        ncOutput.createDimension('x', len(x))
        ncOutput.createDimension('y', len(y))
        ncOutput.createDimension('Layer', eta.shape[0] - 1)
        ncOutput.createDimension('interface', eta.shape[0])
        ncX = ncOutput.createVariable('x', 'float', ('x',))
        ncX[:] = x
        ncY = ncOutput.createVariable('y', 'float', ('y',))
        ncY[:] = y
        ncL = ncOutput.createVariable('Layer', 'float', ('Layer',))
        ncL[:] = layerdensities
        ncEta = ncOutput.createVariable('eta', 'float', ('interface', 'y', 'x'))
        ncEta[:] = eta
        ncTopo = ncOutput.createVariable('topo', 'float', ('y', 'x'))
        ncTopo[:] = -eta[-1]
        ncOutput.close()
    # wind stress
    
    if savewind == True:
        windfilename = outfolder+'/storm_windstress.nc'
        ncOutput = Dataset(windfilename, 'w',format="NETCDF4")
        ncOutput.createDimension('x', eta.shape[2])
        ncOutput.createDimension('y', eta.shape[1])
        ncOutput.createDimension('Time', size = None)
        ncX = ncOutput.createVariable('x', 'float', ('x', ))
        ncX[:] = x
        ncY = ncOutput.createVariable('y', 'float', ('y', ))
        ncY[:] = y
        nctime = ncOutput.createVariable('Time', 'float', ('Time', ))
        nctime[:] = STRESS_X.shape[0]
        ncSX = ncOutput.createVariable('STRESS_X', 'float', ('Time','y','x'))
        ncSX[:] = STRESS_X
        ncSY = ncOutput.createVariable('STRESS_Y', 'float', ('Time','y', 'x'))
        ncSY[:] = STRESS_Y
        ncOutput.close()
    return

def windstress_gaussian(forcing_latwidth = 600,duration = 5,strength = 1,nx=1000,ny=1000,stationary_period = 60,forcing_lonwidth = None,reverse = False):
    ## Duration is a sin**2 function that lasts exactly as long as the duration. Spatially, Gaussian with width marking 3 Std Devs from centre
    
    
    days_forcing = 1
    stationary_period = 60 #minutes
    forcing_period = 12 * 60 #minutes. Force for 12hrs even if duration is much smaller for simplicity. 
    npoints = (stationary_period + forcing_period + (60 - stationary_period)) // 2 #since there's a point every two minutes
    points_per_hour = 30
    hours_forcing = npoints / points_per_hour
    # windstress = np.real(T_sin2(np.linspace(0,hours_forcing,npoints),h = h,offset = stationary_period / 60))   # Inputs are in hours
    # windstress[npoints - 15:] = 0
    
    tgrid = np.linspace(0,hours_forcing,npoints)
    offset = stationary_period / 60
    windstress = 0.5 * np.sin(np.pi * (tgrid - offset)/duration)**2

    windstress *= (offset < tgrid) # remove forcing during the offset (at rest) period
    windstress *= (offset + duration > tgrid) # remove forcing during after forcing has finished (don't let sinusoid start again)
    if reverse == True:
        windstress2 = -0.5 * np.sin(np.pi * (tgrid - offset - duration)/duration)**2

        windstress2 *= (duration + offset < tgrid) # remove forcing during the offset (at rest) period
        windstress2 *= (offset + 2 * duration > tgrid) # remove forcing during after forcing has finished (don't let sinusoid start again)
        windstress += windstress2
    ## Spread forcing over domain with decaying exponential towards boundaries
    STRESS_X = np.zeros([npoints,ny, nx])
    STRESS_Y = np.zeros([npoints,ny, nx])

    ## Altered loop below so that forcing starts and finishes at forcing_latwidth
    ylower = ny//2 - forcing_latwidth
    yupper = ny//2 + forcing_latwidth
    
    ## Modification Feb 20th: For OBCs we don't want forcing right up to the boundary because this gives weird edge effects. Want to taper off 
    ## the meridional forcing near the edges
    if forcing_lonwidth == None:
        for t in range(npoints):
            for j in range(ny):
                STRESS_X[t,j,:] = np.exp((-0.5 * (ny//2 - j - 0.5)**2)/((0.25 * 0.33 * forcing_latwidth)**2)) * windstress[t]
            
            
    return strength * STRESS_X,STRESS_Y


def windstress_cos(forcing_latwidth = 600,duration = 5,strength = 1,nx=1000,ny=1000,stationary_period = 60):
    ## Duration is a sin**2 function that lasts exactly as long as the duration. Spatially, Gaussian with width marking 3 Std Devs from centre
    
    
    
    
    days_forcing = 1
    stationary_period = 60 #minutes
    forcing_period = 12 * 60 #minutes. Force for 12hrs even if duration is much smaller for simplicity. 
    npoints = (stationary_period + forcing_period + (60 - stationary_period)) // 2 #since there's a point every two minutes
    points_per_hour = 30
    hours_forcing = npoints / points_per_hour
    # windstress = np.real(T_sin2(np.linspace(0,hours_forcing,npoints),h = h,offset = stationary_period / 60))   # Inputs are in hours
    # windstress[npoints - 15:] = 0
    
    tgrid = np.linspace(0,hours_forcing,npoints)
    offset = stationary_period / 60
    windstress = 0.5 * np.sin(np.pi * (tgrid - offset)/duration)**2

    windstress *= (offset < tgrid) # remove forcing during the offset (at rest) period
    windstress *= (offset + duration > tgrid) # remove forcing during after forcing has finished (don't let sinusoid start again)

    ## Spread forcing over domain with decaying exponential towards boundaries
    STRESS_X = np.zeros([npoints,ny, nx])
    STRESS_Y = np.zeros([npoints,ny, nx])

    ## Altered loop below so that forcing starts and finishes at forcing_latwidth
    ylower = ny//2 - forcing_latwidth
    yupper = ny//2 + forcing_latwidth
    
    for t in range(npoints):
        for j in range(ny):
            STRESS_X[t,j,:] = np.cos(np.pi * (ny//2 - j) / ny) * windstress[t]
            
            
    return strength * STRESS_X,STRESS_Y


def eta_gaussian(height=500,width=12.5,nx=1000,ny=1000,nlayers=5,H=4000):
    hbottom = np.zeros((ny, nx))
    restintefaceheights = np.linspace(0,-1 * (H - H/nlayers),nlayers)
    temp = height * np.exp(-1 * ((np.arange(nx)) - nx//2)**2/(2 * width)) 
    hbottom = np.broadcast_to(temp , (ny,nx))

    eta = np.zeros([nlayers + 1, ny, nx])
    for j in np.arange(1, nlayers):
        for leny in np.arange(0,ny):
            eta[j, leny, :] =  1*restintefaceheights[j] 


    eta[nlayers,:,:] = - H + hbottom
       
    return eta

def eta_gaussian_ridge(height=500,width=12.5,nx=1000,ny=1000,nlayers=5,H=4000):
    hbottom = np.zeros((ny, nx))
    restintefaceheights = np.linspace(0,-1 * (H - H/nlayers),nlayers)
    
    x,y = np.meshgrid(np.arange(nx),np.arange(ny))
    
    hbottom = height * np.exp(-1 * (x - nx//2)**2/(2 * width))
    eta = np.zeros([nlayers + 1, ny, nx])
    for j in np.arange(1, nlayers):
        for leny in np.arange(0,ny):
            eta[j, leny, :] =  1*restintefaceheights[j] 


    eta[nlayers,:,:] = - H + hbottom
    
    return eta


def eta_gaussian_hill(height=500,width=12.5,nx=1000,ny=1000,nlayers=5,H=4000,ridge = False):
    hbottom = np.zeros((ny, nx))
    restintefaceheights = np.linspace(0,-1 * (H - H/nlayers),nlayers)
    
    x,y = np.meshgrid(np.arange(nx),np.arange(ny))
    
    hbottom = height * np.exp(-1 * (x - nx//2)**2/(2 * width)) 
    
    if ridge == False:
        hbottom *= np.exp(-1 * (y - ny//2)**2/(2 * width))
    
    eta = np.zeros([nlayers + 1, ny, nx])
    for j in np.arange(1, nlayers):
        for leny in np.arange(0,ny):
            eta[j, leny, :] =  1*restintefaceheights[j] 


    eta[nlayers,:,:] = - H + hbottom
       
    return eta

def calculate_pressure_new(e,densities):
    nlayers = e.zi.shape[0] - 1
    
    e_0 = e.isel(time = 0)

    e_anom = e - e_0
    
    ref_density = densities.isel(zl = 0)
    anom_densities = densities - ref_density
    
    pressure_anomaly = e.isel(zi = slice(0,nlayers)) * 0

    pressure_anomaly[:,0,:,:] =  9.8 * densities.isel(zl = 0) * (e.isel(zi = 0))

    for i in range(1,nlayers):
        pressure_anomaly[:,i,::] = 9.8 * (anom_densities.isel(zl = i)) * (e_anom.isel(zi = i))

    pressure = pressure_anomaly.cumsum("zi")
    
    pressure = xr.DataArray(
        data = pressure,
        dims = ["time","zl","yh","xh"],
        coords = {"time":e.time.values,"zl":densities.zl.values,"yh":e.yh.values,"xh":e.xh.values})
    return pressure

def calculate_pressure(e,densities):
    nlayers = e.zi.shape[0] - 1
    
    e_0 = e.isel(time = 0)

    e_anom = e - e_0
    
    ref_density = densities.isel(zl = 0)
    anom_densities = densities - ref_density
    
    pressure_anomaly = e.isel(zi = slice(0,nlayers)) * 0

    pressure_anomaly[:,0,:,:] = -1 * 9.8 * densities.isel(zl = 0) * (e.isel(zi = 0))

    for i in range(1,nlayers):
        pressure_anomaly[:,i,::] = 9.8 * (anom_densities.isel(zl = i)) * (e_anom.isel(zi = i + 1) - e_anom.isel(zi = i))

    pressure = pressure_anomaly.cumsum("zi")
    
    pressure = xr.DataArray(
        data = pressure,
        dims = ["time","zl","yh","xh"],
        coords = {"time":e.time.values,"zl":densities.zl.values,"yh":e.yh.values,"xh":e.xh.values})
    return pressure

def sup_filter(field):
    FIELD = xrft.fft(field.load(),dim = "time")
    FIELD_filtered = FIELD.where(np.abs(FIELD.freq_time) > f,0)
    return np.real(xrft.ifft(FIELD_filtered,dim = "freq_time"))

def process(experiment,tlim = None,ymax = 625,ymin = 375,outpath = None):
    if outpath == None:
        outpath = experiment
    savepath = "/g/data/v45/ab8992/processed_data/automated/" + outpath 


    out_no = "output00*"
    data = {}
    inputfolder = "/home/149/ab8992/bottom_near_inertial_waves/automated/" + experiment + "/archive/" + out_no
    prog_hourly = xr.open_mfdataset(inputfolder + "/prog_hourly.nc*", decode_times=False, engine='netcdf4',parallel = True,chunks = "auto")
    prog_hourly = prog_hourly.isel(yh = slice(ymin,ymax),yq = slice(ymin,ymax),time = slice(0,tlim))
    e = prog_hourly.e.astype(float)
    v = prog_hourly.v.astype(float)
    u = prog_hourly.u.astype(float)
    
    
    densities = prog_hourly.zl.load()
    tau = prog_hourly.taux.astype(float)
    v_interp = v.interp(yq = u.yh).drop("yq").isel(time = slice(0,tlim))
    u_interp = u.interp(xq = v.xh).drop("xq").isel(time = slice(0,tlim))
    
    u.where(u.xh**2 + u.yh**2 > 2, 0)
    
    w = ((u_interp.differentiate("xh") + v_interp.differentiate("yh")).isel(zl = -1) * (e.isel(zi = -1) - e.isel(zi = -2).isel(time = 0)))    
       
    
    tau = tau.interp(xq = v.xh).drop("xq").isel(time = slice(0,tlim))
    pressure = calculate_pressure_new(e,densities)
    N = np.sqrt(((u.zl.values[-1] - u.zl.values[0]) * 9.8) / (3275 * 1027.81))  # N in s^-1
    
    data = xr.Dataset(
        data_vars = {"u":u_interp,
                     "v":v_interp,
                     "tau":tau,
                     "densities":densities,
                     "phi":pressure,
                     "e":e,
                     "w":w,
                     "N":N
                    }
    )
    data_SIunits = data.assign_coords({"time":("time",data.time.values * 3600,data.time.attrs),"xh":("xh",data.xh.values * 1000,data.xh.attrs),"yh":("yh",data.yh.values * 1000,data.yh.attrs)})
    data_SIunits.time.attrs["units"] = "seconds since 0001-01-01 00:00:00"
    data_SIunits.xh.attrs["units"] = "meters"
    data_SIunits.yh.attrs["units"] = "meters"
    
    
    
    

    try:
        os.remove(savepath)
    except:
        pass

    data_SIunits.isel(xh = slice(1,None),yh = slice(1,None)).to_netcdf(savepath)
    return

def run_mom6(experiment):
    subprocess.run(["/home/149/ab8992/bottom_near_inertial_waves/automated/runscript"],cwd = "/home/149/ab8992/bottom_near_inertial_waves/automated/" + experiment)
    return

def set_noforcing(experiment):
    subprocess.run(["/home/149/ab8992/bottom_near_inertial_waves/automated/forcing_off.sh"],cwd = "/home/149/ab8992/bottom_near_inertial_waves/automated/" + experiment)
    return
# setup 


def run_lagrange_filtering(experiment,xextent = [None,None],yextent = [None,None],nlayers = 5,outname = ""):
    
    if outname != "":
        outputdir = "/g/data/v45/ab8992/processed_data/automated/lagrangefilter/" + outname
        
    else:
        outputdir = "/g/data/v45/ab8992/processed_data/automated/lagrangefilter/" + experiment
        
    inputdir = "/g/data/v45/ab8992/processed_data/automated/" + experiment

    script = f"""
import filtering
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import os
import sys

inputdir = "{inputdir}"
outputdir = "{outputdir}"
xextent = [{xextent[0]},{xextent[1]}]
yextent = [{yextent[0]},{yextent[1]}]
nlayers = {nlayers}


variables = {{"U": "u", "V": "v","phi":"phi","w":"w"}}
filenames = {{v: inputdir for v in variables.keys()}}
dimensions = {{"lon": "xh", "lat": "yh", "time": "time", "depth": "zl"}}

for i in range(nlayers): 

    if os.path.exists(outputdir + "_" + str(i) + ".nc"):
        os.remove(outputdir + "_" + str(i) + ".nc")
    print(i)
    f = filtering.LagrangeFilter(outputdir + "_" + str(i), filenames, variables, dimensions,
        sample_variables=["U","V","phi","w"], 
        mesh="flat",
        highpass_frequency = 9e-5,
        advection_dt =timedelta(minutes=5),
        window_size=timedelta(days=2).total_seconds(),indices={{"depth": [i]}}
    )

    f.seed_subdomain(min_lat=yextent[0], max_lat=yextent[1],min_lon = xextent[0],max_lon = xextent[1])

    f.make_zonally_periodic()
    f()
    """ 
    file = open(f"/home/149/ab8992/bottom_near_inertial_waves/automated/{experiment}/lfilter.py","w")
    file.writelines(script)
    subprocess.run(
        "qsub /home/149/ab8992/bottom_near_inertial_waves/lagrange_filtering_scripts/filter_script.sh",
        cwd = f"/home/149/ab8992/bottom_near_inertial_waves/automated/{experiment}",
        shell=True
    )
    return

def montgomery(e,densities):
    """
    Calculates the montgomery pressure given interface heights and densities
    """
    eprime = (e - e.isel(time = 0) )
    g = 9.8

    Mg = (e.isel(zi = slice(1,None)) * 0).rename({"zi":"zl"}).assign_coords({"zl":densities.values})
    Mg[:,0,:,:] = e.isel(zi = 0) * g ## Calculate the first value
    for i in range(1,Mg.shape[1]):
        Mg[:,i,:,:] = Mg[:,i-1,:,:] + eprime.isel(zi = i) * (densities[i] - densities[i-1])*g / (densities[i-1])

    return Mg

def get_thickness(e,densities):
    """
    Given the interface heights, calulate the layer thicknesses
    """

    thickness = e.isel(zi = slice(0,e.zi.shape[0] - 1)) * 0

    for i in range(thickness.zi.shape[0]):
        thickness[:,i,:,:] = (e.isel(zi = i) - e.isel(zi = i+1)).values  

    thickness = thickness.rename({"zi":"zl"}).assign_coords({"zl":densities.values})
    return thickness


def dwa(thickness,field):

    H = thickness.sum("zl")


    return (field * thickness).sum("zl") / H


def barocl_ef(e,u,densities):
    """
    Calculates the baroclinic EF in the method of Simmons et al
    """

    thickness = get_thickness(e,densities)

    ubar = dwa(thickness,u)
    uprime = u - ubar
    Mg = montgomery(e,densities)
    # Mgbar,h = dwa(array,array.u)
    Mgbar = dwa(thickness,Mg)

    Mgprime = Mg - Mgbar

    EF = (Mgprime * thickness * uprime)

    return EF


def process_baroclinic_ef(experiment,tlim = None,ymax = 625,ymin = 375,xmax = 625,xmin=375,outpath = None):
    """
    it's become apparent that calculating the EF via a thickness weighted average of Montgomery's Potential and u is the way to go. I don't need to calculate pressure or w any more, but storing the 
    Montgomery Potential and the twa velocities will be helpful
    """
    if outpath == None:
        outpath = experiment
    savepath = "/g/data/v45/ab8992/processed_data/automated/simmonsmethod/" + outpath 


    out_no = "output00*"
    data = {}
    inputfolder = "/home/149/ab8992/bottom_near_inertial_waves/automated/" + experiment + "/archive/" + out_no
    prog_hourly = xr.open_mfdataset(inputfolder + "/prog_5min.nc*", decode_times=False, engine='netcdf4',parallel = True,chunks = "auto")
    prog_hourly = prog_hourly.isel(yh = slice(ymin,ymax),yq = slice(ymin,ymax),xh = slice(xmin,xmax),xq = slice(xmin,xmax),time = slice(0,tlim))
    e = prog_hourly.e.astype(float)
    v = prog_hourly.v.astype(float)
    u = prog_hourly.u.astype(float)
    
    
    densities = prog_hourly.zl.load()
    tau = prog_hourly.taux.astype(float)
    v_interp = v.interp(yq = u.yh).drop("yq").isel(time = slice(0,tlim))
    u_interp = u.interp(xq = v.xh).drop("xq").isel(time = slice(0,tlim))
    
      
    
    tau = tau.interp(xq = v.xh).drop("xq").isel(time = slice(0,tlim))

    thickness = get_thickness(e,densities)
    u_dwa = dwa(thickness,u_interp)
    v_dwa = dwa(thickness,v_interp)
    Mg = montgomery(e,densities)
    Mgprime = Mg - dwa(thickness,Mg)

    uprime = u_interp - u_dwa
    vprime = v_interp - v_dwa

    EF_zonal = (Mgprime * uprime * thickness).sum("zl")
    EF_merid = (Mgprime * vprime * thickness).sum("zl")

    data = xr.Dataset(
        data_vars = {"u":u_interp,
                     "v":v_interp,
                     "e":e,
                     "tau":tau,
                     "densities":densities,
                     "EF_zonal":EF_zonal,
                     "EF_merid":EF_merid,
                    }
    )
    data_SIunits = data.assign_coords({"time":("time",data.time.values * 3600,data.time.attrs),"xh":("xh",data.xh.values * 1000,data.xh.attrs),"yh":("yh",data.yh.values * 1000,data.yh.attrs)})
    data_SIunits.time.attrs["units"] = "seconds since 0001-01-01 00:00:00"
    data_SIunits.xh.attrs["units"] = "meters"
    data_SIunits.yh.attrs["units"] = "meters"
    

    try:
        os.remove(savepath)
    except:
        pass

    data_SIunits.isel(xh = slice(1,None),yh = slice(1,None)).to_netcdf(savepath)
    return


def process_slices(experiment,tlim = None,ymax = 625,ymin = 375,xmax = 625,xmin=375,outpath = None):
    """
    Copy of the baroclinic EF function in autolib. This time just return the fluxes along the boundaries of provided domain rather than 
    calculate huge array of EF
    """
    if outpath == None:
        outpath = experiment
    savepath = "/g/data/v45/ab8992/processed_data/automated/simmonsmethod/" + outpath 

    out_no = "output00*"
    data = {}
    inputfolder = "/home/149/ab8992/bottom_near_inertial_waves/automated/" + experiment + "/archive/" + out_no
    prog_hourly = xr.open_mfdataset(
        inputfolder + "/prog_hourly.nc*", decode_times=False, engine='netcdf4',parallel = True,chunks = "auto"
        )
    prog_hourly = prog_hourly.isel(
        yh = slice(ymin,ymax),yq = slice(ymin,ymax),xh = slice(xmin,xmax),xq = slice(xmin,xmax),time = slice(0,tlim)
        )
    
    prog_hourly = prog_hourly.assign_coords({"time":("time",prog_hourly.time.values * 3600,prog_hourly.time.attrs),
                                       "xh":("xh",prog_hourly.xh.values * 1000,prog_hourly.xh.attrs),
                                       "yh":("yh",prog_hourly.yh.values * 1000,prog_hourly.yh.attrs),
                                       "xq":("xq",prog_hourly.xq.values * 1000,prog_hourly.xq.attrs),
                                       "yq":("yq",prog_hourly.yq.values * 1000,prog_hourly.yq.attrs),
                                       })


    e = prog_hourly.e.astype(float)
    v = prog_hourly.v.astype(float)
    u = prog_hourly.u.astype(float)
    
    
    densities = prog_hourly.zl.load()
    tau = prog_hourly.taux.astype(float)
    v_interp = v.interp(yq = u.yh).drop("yq").isel(time = slice(0,tlim))
    u_interp = u.interp(xq = v.xh).drop("xq").isel(time = slice(0,tlim))
    tau = tau.interp(xq = v.xh).drop("xq")
    
    uprime = u_interp - u_dwa
    vprime = v_interp - v_dwa

    wind_work = (tau * u_interp.isel(zl = 0)).isel(
        time = slice(0,20),
        xh = slice(0,5)
        ).integrate("time").integrate("xh")

    EF_zonal = (1027 * Mgprime * uprime * thickness).sum("zl").isel(xh = [0,-2]).rename({"xh":"xi"})
    EF_merid = (1027 * Mgprime * vprime * thickness).sum("zl").isel(yh = [0,-2]).rename({"yh":"yi"})

    data = xr.Dataset(
        data_vars = {
                     "EF_zonal":EF_zonal,
                     "EF_merid":EF_merid,
                     "wind_work":wind_work
                    }
    )
    # data_SIunits = data.assign_coords({"time":("time",data.time.values * 3600,data.time.attrs),
    #                                    "xh":("xh",data.xh.values * 1000,data.xh.attrs),
    #                                    "yh":("yh",data.yh.values * 1000,data.yh.attrs),
    #                                    "yi:":("yi",np.array([0,1]),None),
    #                                    "xi:":("xi",np.array([0,1]),None)
                                    #    })
    data.time.attrs["units"] = "seconds since 0001-01-01 00:00:00"
    data.xh.attrs["units"] = "meters"
    data.yh.attrs["units"] = "meters"
    
    if outpath != None:
        data.isel(xh = slice(1,None),yh = slice(1,None)).to_netcdf(savepath)


    return data