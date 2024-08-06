from matplotlib import rc
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import os
import xarray as xr
import subprocess
import matplotlib.pyplot as plt
import shutil
from pathlib import Path
import xrft

basepath = Path.cwd().absolute()


def setup_mom6(exptname,perturbation,overrides = [],walltime = None,default_dir = "default",run_duration = False):
    ## If common forcing is provided, set another input folder that contains the windstress for all runs in this experiment
    default_dir = basepath / "rundirs" / default_dir
    perturbation = str(perturbation)
    runpath = basepath / "rundirs" / exptname / f"{exptname}_{perturbation}"
    runpath.mkdir(parents = True,exist_ok = True)

    subprocess.run(f"cp {default_dir}/* {runpath} -r",shell = True)

    ## We've created the new directory with name 'name'
    # Now need to edit config file to point to the name of the topography configuraation 'tname'
    file = open(f"{runpath}/config.yaml","r")
    config = file.readlines()
    file.close()
    ##Iterate through and find jobname and input
    
    for line in range(len(config)):
        if "jobname" in config[line][0:10]:
            config[line] = "jobname: " + exptname + "_" + perturbation + "\n"
            
        if "input" in config[line][0:10]:
            if perturbation != "common":
                config[line] = f"input:\n    - {basepath / 'inputdir' / (exptname + '_' + perturbation)}\n    - {basepath / 'inputdir' / 'common'}\n"
            else:
                config[line] = f"input:\n    - {basepath / 'inputdir' / 'common'}\n"
        if "walltime" in config[line][0:10] and walltime != None:
            config[line] = "walltime: " + str(walltime)
    file = open(runpath / "config.yaml","w")
    file.writelines(config)
    file.close()
    
    ## Update override file
    
    file = open(runpath / "MOM_override","r")
    override_file = file.readlines()
    file.close()
    
    for i in overrides:
        override_file.append("#override " + i + str("\n"))
        
    file = open(runpath / "MOM_override","w")
    file.writelines(override_file)


    
    return

def save_inputdata(x,y,STRESS_X,STRESS_Y,eta,tname,savewind =True,strat = 1,savedensities = True,**kwargs):
   
    outfolder = basepath / "inputdir" / tname
    
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
        
        
    else:
        if "driver-layers-topo-densities.nc" in os.listdir(outfolder):
            os.remove(outfolder / "driver-layers-topo-densities.nc")
        if "storm_windstress.nc" in os.listdir(outfolder):
            os.remove(outfolder / "storm_windstress.nc")
    
    if savedensities == True:
        ## Create default layer densities for this number of layers
        nlayers = eta.shape[0] - 1

        drho_tot = 4 ## Total change in salinity from surface to bottom

        drho_eff = (drho_tot * (nlayers - 1) / nlayers) / 2

        rho_ref = 1027
        layerdensities = np.linspace(rho_ref - drho_eff * strat,rho_ref + drho_eff * strat,nlayers)
        driverfilename = outfolder / 'driver-layers-topo-densities.nc'
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
        windfilename = outfolder / 'storm_windstress.nc'
        ncOutput = Dataset(windfilename, 'w',format="NETCDF4")
        ncOutput.createDimension('x', eta.shape[2])
        ncOutput.createDimension('y', eta.shape[1])
        ncOutput.createDimension('Time', size = None)
        ncX = ncOutput.createVariable('x', 'float', ('x', ))
        ncX[:] = x
        ncY = ncOutput.createVariable('y', 'float', ('y', ))
        ncY[:] = y
        nctime = ncOutput.createVariable('Time', 'float', ('Time', ))
        nctime[:] = np.arange(0,STRESS_X.shape[0] * 2,2)
        ncSX = ncOutput.createVariable('STRESS_X', 'float', ('Time','y','x'))
        ncSX[:] = STRESS_X
        ncSY = ncOutput.createVariable('STRESS_Y', 'float', ('Time','y', 'x'))
        ncSY[:] = STRESS_Y
        ncOutput.close()
    return

def windstress_gaussian(forcing_latwidth = 100,duration = 5,strength = 1,nx=1000,ny=1000,forcing_lonwidth = None,reverse = True,gridspacing = 2,**kwargs):
    ## Duration is a sin**2 function that lasts exactly as long as the duration. Spatially, Gaussian with width marking 3 Std Devs from centre
    
    ## MODIFICATION 24/7/23
    # When altering duration, need to change strength to include effect of increasing total energy in. This is linear - simply scale strength by deviation of duration from 5

    forcing_latwidth = forcing_latwidth // gridspacing

    if duration != 5:
        strength *= 5 / duration
    #################################

    forcing_period = (duration *2) * 60 #minutes. Force for 12hrs even if duration is much smaller for simplicity. 
    # npoints = (stationary_period + forcing_period + (60 - stationary_period)) // 2 #since there's a point every two minutes
    npoints = int((forcing_period ) // 5)  #since there's a point every 5 minutes
    points_per_hour = 30
    hours_forcing = npoints / points_per_hour
    # windstress = np.real(T_sin2(np.linspace(0,hours_forcing,npoints),h = h,offset = stationary_period / 60))   # Inputs are in hours
    # windstress[npoints - 15:] = 0
    
    tgrid = np.linspace(0,hours_forcing,npoints)
    offset = 0 # Redundant now
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

    ## Altered loop below so that forcing starts and finishes at forcing_latwidth
    ylower = ny//2 - forcing_latwidth
    yupper = ny//2 + forcing_latwidth
    

    for t in range(npoints):
        for j in range(ny):
            STRESS_X[t,j,:] = np.exp(- (
                ((ny//2 - j - 0.5) * 2) / forcing_latwidth
                )**2 
            )* windstress[t]
            
            
    return strength * STRESS_X

def eta_gaussian_hill(height=500,width=12.5,nx=1000,ny=1000,nlayers=5,H=4000,ridge = True,gridspacing = 2,**kwargs):
    width /= gridspacing
    hbottom = np.zeros((ny, nx))
    restintefaceheights = np.linspace(0,-1 * (H - H/nlayers),nlayers)
    
    x,y = np.meshgrid(np.arange(nx),np.arange(ny))
    
    hbottom = height * np.exp(- ( 
                               (2 * (x - nx//2)) /  width
                               )**2
    ) 

    if ridge == False:
        hbottom *= np.exp(-1 * (y - ny//2)**2/(2 * width))
    
    eta = np.zeros([nlayers + 1, ny, nx])
    for j in np.arange(1, nlayers):
        for leny in np.arange(0,ny):
            eta[j, leny, :] =  1*restintefaceheights[j] 


    eta[nlayers,:,:] = - H + hbottom
       
    return eta

def calculate_pressure(e,densities,pressure_type = "total"):
    nlayers = e.zi.shape[0] - 1
    
    e_0 = e.isel(time = 0)

    e_anom = e - e_0
    
    ref_density = densities.isel(zl = 0)
    anom_densities = densities - ref_density
    
    pressure_anomaly = e.isel(zi = slice(0,nlayers)) * 0

    if "xh" in e.dims:
        pressure_anomaly[:,0,:] =  9.8 * densities.isel(zl = 0) * (e.isel(zi = 0))
        for i in range(1,nlayers):
            pressure_anomaly[:,i,:] = 9.8 * (anom_densities.isel(zl = i)) * (e_anom.isel(zi = i))

    else:
        pressure_anomaly[:,0] =  9.8 * densities.isel(zl = 0) * (e.isel(zi = 0))
        for i in range(1,nlayers):
            pressure_anomaly[:,i] = 9.8 * (anom_densities.isel(zl = i)) * (e_anom.isel(zi = i))
            

    # pressure = pressure_anomaly.cumsum("zi")
    # pressure = pressure_anomaly.sum("zi")
    
    pressure = xr.DataArray(
        data = pressure_anomaly.cumsum("zi"),
        dims = e.rename({"zi":"zl"}).dims,
        # dims = ["time","zl","xh"],
        # coords = {"time":e.time.values,"zl":densities.zl.values,"xh":e.xh.values})
        coords = e.drop("zi").coords)    
    
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
    pressure = calculate_pressure(e,densities)
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
    subprocess.run("bash /home/149/ab8992/bottom_near_inertial_waves/automated/runscript &",cwd = "/home/149/ab8992/bottom_near_inertial_waves/revision/" + experiment,shell = True)
    return

def set_noforcing(experiment):
    subprocess.run(["/home/149/ab8992/bottom_near_inertial_waves/automated/forcing_off.sh"],cwd = "/home/149/ab8992/bottom_near_inertial_waves/" + experiment)
    return
# setup 



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


### Experiment Class

## Iterate over topog type and nlayers
from importlib import reload
# reload(al)

def set_si_coords(array):
    return array.assign_coords({"time":("time",array.time.values * 60),
                             "xq":("xq",array.xq.values * 1000),
                             "xh":("xh",array.xh.values * 1000),
                             "yh":("yh",array.yh.values * 1000),
                             "yq":("yq",array.yq.values * 1000)
    })



class expt:
    def __init__(self,x,y,nlayers,variable,var_value,overrides = []):
        self.nlayers = nlayers
        self.variable = variable
        self.var_value = var_value
        self.x = x
        self.y = y
        self.overrides = overrides
        self.path = basepath / "rundirs" / self.variable / f"{self.variable}_{self.var_value}"

    def make_inputs(self):
    ## First, determine what needs to be generated. Wind, or topo    

        if self.variable in ["height","topo_width","strat"]:
            ## Make topog file
            eta = eta_gaussian_hill(
                nlayers=self.nlayers,
                nx = len(self.x),
                ny = len(self.y),
                **{self.variable : self.var_value}
                )
            save_inputdata(self.x,
                 self.y,
                 None,
                 None,
                 eta,
                 f"{self.variable}_{self.var_value}",
                 savewind = False,
                 **{self.variable : self.var_value}
                 )
            
        else:
            ## Make and save wind stress
            STRESS_X = windstress_gaussian(nx = len(self.x),
                                           ny = len(self.y),
                                           **{self.variable : self.var_value})
            save_inputdata(self.x,
                self.y,
                STRESS_X,
                STRESS_X * 0,
                np.zeros((1,len(self.y),len(self.x))),
                f"{self.variable}_{self.var_value}",
                savedensities = False
                )
        return
    
    def setup(self,overrides = None,walltime = None,default_dir = None,run_duration = 10,forcing_path=None):
        setup_mom6(self.variable,self.var_value,overrides = self.overrides)
        return

    def run(self,fast = False):
        print("RUNNING: " + self.path.name)
        if not fast:
            subprocess.run(
            f"payu setup -f",shell= True,cwd = str(self.path)
            )

        subprocess.run(
           f"payu run -f",shell= True,cwd = str(self.path)
        )
        return

    def load(self):
        zonal = xr.open_mfdataset([i for i in (basepath / "outputdir" / f"{self.variable}_{self.var_value}" / "zonal").glob("*.nc")],decode_times = False,parallel = True)
        merid = xr.open_mfdataset([i for i in (basepath / "outputdir" / f"{self.variable}_{self.var_value}" / "merid").glob("*.nc")],decode_times = False,parallel = True)
        return {"zonal":zonal,"merid":merid}


    def process_output(self,xrange = [-100,100], yrange = [-1510,-1490],tlim = None,outpath = None,integrate = True):
        """
        Copy of the baroclinic EF function in autolib. This time just return the fluxes along the boundaries of provided domain rather than 
        calculate huge array of EF
        """
        datapath = f"/home/149/ab8992/bottom_near_inertial_waves/{expt_suite_name}/{self.exptname}/{self.runname}"

        e = xr.open_mfdataset(datapath +  "/archive/output00*/e_10min.nc",decode_times = False,parallel = True,chunks = {"zl":1,"time":50}).sel(
            xh = slice(xrange[0],xrange[1]),yh = slice(yrange[0],yrange[1])).isel(
                time = slice(0,tlim)
            )
        u = xr.open_mfdataset(datapath +  "/archive/output00*/u_10min.nc",decode_times = False,parallel = True,chunks = {"zl":1,"time":50}).sel(
            xq = slice(xrange[0],xrange[1]),yh = slice(yrange[0],yrange[1])).isel(
                time = slice(0,tlim)
            )
        # v = xr.open_mfdataset(datapath +  "/archive/output00*/v_10min.nc").sel(
        #     xh = slice(xrange[0],xrange[1]),yh = slice(yrange[0],yrange[1])).isel(
        #         time = slice(0,tlim)
        #     )
        tau = xr.open_mfdataset(datapath +  "/archive/output00*/taux_10min.nc",decode_times = False,parallel = True,chunks = {"time":50}).sel(
            xq = slice(xrange[0],xrange[1]),yh = slice(yrange[0],yrange[1])).isel(
                time = slice(0,tlim)
            )

        # v = data.v.interp(yq = data.u.yh).drop("yq").sel(xh = slice(xrange[0],xrange[1]),yh = slice(yrange[0],yrange[1]))
        e = e.e.assign_coords({"time":("time",e.time.values * 60),
                                "xh":("xh",e.xh.values * 1000),
                                "yh":("yh",e.yh.values * 1000)
            }).astype(float)
        tau = tau.taux.assign_coords({"time":("time",tau.time.values * 60),
                                "xq":("xq",tau.xq.values * 1000),
                                "yh":("yh",tau.yh.values * 1000)
            }).interp(xq = e.xh).drop("xq").astype(float)
        u = u.u.assign_coords({"time":("time",u.time.values * 60),
                                "xq":("xq",u.xq.values * 1000),
                                "yh":("yh",u.yh.values * 1000)
            }).interp(xq = e.xh).drop("xq").astype(float)


        # e = (e).persist()
        u = u.persist()
        # tau = tau.persist()

        # densities = u.zl.load()
        # thickness = al.get_thickness(e,densities)
        # u_dwa = al.dwa(thickness,u)
        # # v_dwa = dwa(thickness,v)
        # Mg = al.montgomery(e,densities)
        # Mgprime = Mg - al.dwa(thickness,Mg)

        # uprime = u - u_dwa
        # vprime = v - v_dwa

        # conv = (1027 * Mgprime.differentiate("xh") * thickness * u_dwa).sum("zl")

        wind_work = (tau * u.isel(zl = 0)).isel(
            time = slice(0,500),
            ).sel(xh = slice(-25000,25000)).integrate("xh").integrate("yh")

        # EF_zonal = (1027 * Mgprime * uprime * thickness).sum("zl").isel(xh = [0,-1]).rename({"xh":"xi"}).integrate("yh")
        # EF_merid = (1027 * Mgprime * vprime * thickness).sum("zl").isel(yh = [0,-1]).rename({"yh":"yi"})

        wind_work = wind_work.integrate("time")
        data = xr.Dataset(
            data_vars = {
                        # "EF_zonal":EF_zonal,
                        "wind_work":wind_work
                        # "u":u,
                        # "e":e,
                        # "thickness":thickness,## Comment these out to return to what I originally output
                        # "Mgprime":Mgprime, 
                        # "uprime":uprime
                        }
        )
        data.wind_work.attrs["units"] = "Integrated over meridional extent and 50km zonally"
        
        data.to_netcdf(f"/g/data/v45/ab8992/bottom_iwbs/{expt_suite_name}/{self.runname}")
        return data
    
    ## If you screw up, run to delete all outputs and start from scratch
    def reset(self):
        try:
            subprocess.run(f"rm /home/149/ab8992/bottom_near_inertial_waves/{expt_suite_name}/{self.exptname}/{self.runname}/archive/* -rf",shell = True)
            subprocess.run(f"rm /home/149/ab8992/bottom_near_inertial_waves/{expt_suite_name}/{self.exptname}/{self.runname}/* -rf",shell = True)
        except:
            print(self.runname)
    
