## Python script for running postprocessing

from matplotlib import rc
import numpy as np
from netCDF4 import Dataset
import xrft
import matplotlib.pyplot as plt
import os
import xarray as xr
import subprocess
import matplotlib.pyplot as plt
import shutil
os.chdir("/home/149/ab8992/bottom_near_inertial_waves/automate_experiments")


## Make this script use argparse to accept a single string artument called expt


import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-e','--expt', type=str,
)
args = parser.parse_args()
expt = args.expt


import autolib as al
from dask.distributed import Client
# client = Client(n_workers=28)
client = Client()


## Iterate over topog type and nlayers
from importlib import reload
expt_suite_name = "revision"

def set_si_coords(array):
    return array.assign_coords({"time":("time",array.time.values * 60),
                             "xq":("xq",array.xq.values * 1000),
                             "xh":("xh",array.xh.values * 1000),
                             "yh":("yh",array.yh.values * 1000),
                             "yq":("yq",array.yq.values * 1000)
    })

def openexpt(pert,values,topo,layers = 2,merid = False):

    axes = {
        "KEtot":[],
        "PEtot":[],
        "windw":[],
        "xax":[]
    }
    for i in values:
        try:
            if merid == True:
                data = xr.open_dataset(f"/g/data/v45/ab8992/bottom_iwbs/august/merid_june_{pert}_{topo}_{layers}layer_{pert}-{i}")

            if merid == False:
                data = xr.open_dataset(f"/g/data/v45/ab8992/bottom_iwbs/august/june_{pert}_{topo}_{layers}layer_{pert}-{i}")
        except:
            print(f"Trouble opening {pert} {i}" )


        axes["KEtot"].append(data.KEtot.sum("zl").values) ## I'd previously multiplied by depth. This is wrong
        axes["PEtot"].append(data.PEtot.sum("zi").values)
        axes["xax"].append(i)
        axes["windw"].append(data.ww_tot.values)
        data.close()
    for i in axes:
        axes[i] = np.array(axes[i])
    return axes



class expt:
    def __init__(self,x,y,nlayers,variable,var_value,topo,common):
        self.nlayers = nlayers
        self.variable = variable
        self.var_value = var_value
        self.topo = topo
        self.common = common
        self.x = x
        self.y = y
        self.runname = f"{self.variable}_{self.topo}_{self.nlayers}layer_{self.variable}-{self.var_value}"
        self.exptname = f"{self.variable}_{self.topo}_{self.nlayers}layer"

        if not os.path.exists(f"/home/149/ab8992/bottom_near_inertial_waves/{expt_suite_name}/{self.exptname}/{self.runname}"):
            os.makedirs(f"/home/149/ab8992/bottom_near_inertial_waves/{expt_suite_name}/{self.exptname}/{self.runname}")

        subprocess.run(f"ln -s /g/data/v45/ab8992/mom6_channel_configs/{expt_suite_name}/{self.exptname}/{self.runname} /home/149/ab8992/bottom_near_inertial_waves/{expt_suite_name}/{self.exptname}/{self.runname}/inputdir",shell=True)
    
    def make_topo(self):
        if self.common == "windforcing":
            ## Make topog file
            ridge = self.topo == "ridge"
            eta = al.eta_gaussian_hill(
                nlayers=self.nlayers,
                ridge=ridge,
                nx = len(self.x),
                ny = len(self.y),
                **{self.variable : self.var_value}
                )

            al.save_topo(self.x,
                 self.y,
                 None,
                 None,
                 eta,
                 f"/{expt_suite_name}/{self.exptname}/{self.runname}",
                 savewind = False,
                 **{self.variable : self.var_value}
                 )
        elif self.common == "topog":
            ## Make and save wind stress
            STRESS_X = al.windstress_gaussian(nx = len(self.x),
                                           ny = len(self.y),
                                           reverse = True,
                                           **{self.variable : self.var_value})
            al.save_topo(self.x,
                self.y,
                STRESS_X,
                STRESS_X * 0,
                np.zeros((1,len(self.y),len(self.x))),
                f"/{expt_suite_name}/{self.exptname}/{self.runname}",
                savedensities = False
                )
        return
    
    def setup(self,overrides = None,walltime = None,default_dir = None,run_duration = 10,forcing_path=None):
        print("SETUP: " + self.runname)

        # june_common_ridge_{nlayers}layers
        if forcing_path == None:
            forcing_path = f"{expt_suite_name}/{self.exptname}/{self.runname}" 

        if self.common == "topog":
            common_forcing_path = f"june_common_{self.topo}_{self.nlayers}layers"
        elif self.common == "windforcing":
            common_forcing_path = f"june_common_wind"
        al.setup_mom6(f"{expt_suite_name}/{self.exptname}/{self.runname}",
                    forcing_path,
                    walltime = walltime,
                    overrides = overrides + [f"NK={self.nlayers}"],
                    common_forcing = f"{expt_suite_name}/common/{common_forcing_path}",
                    default_dir=default_dir,
                    run_duration = run_duration
                    )   

    def run(self):
        print("RUNNING: " + self.runname)
        al.run_mom6(f"{self.exptname}/{self.runname}")   
        return
    def set_noforcing(self):
        subprocess.run(
            "/home/149/ab8992/bottom_near_inertial_waves/automated/forcing_off.sh",
            cwd = f"/home/149/ab8992/bottom_near_inertial_waves/{expt_suite_name}/{self.exptname}/{self.runname}")        
        return
    def fastrun(self):
        subprocess.run(
            "payu run -f",shell= True,cwd = f"/home/149/ab8992/bottom_near_inertial_waves/{expt_suite_name}/{self.exptname}/{self.runname}"
        )
        return


    def process_output(self,merid = False,basepath = "/g/data/v45/ab8992/bottom_iwbs/",foldername = "august"):
        

        xh = np.arange(-1999,2000,2.)
        yh = np.arange(-1999,-1000,2)


        if merid == False:
            outpath = basepath +  "/" + foldername + "/" + self.runname ## Modification to allow function to work as class

            dim = "xh"
            otherdim = "yh"
            sliceint = 250
            eslice = xr.open_mfdataset(
                f"/home/149/ab8992/bottom_near_inertial_waves/{expt_suite_name}/{self.exptname}/{self.runname}/archive/output00*/e_10min.nc",chunks = {"zi":1},decode_times = False).isel({otherdim : sliceint,"time" : slice(0,4152//2)}).isel(zi = slice(1,5))

            #! Warning, this is not interpolating correctly. Hardcoded for zonal slice only, selects middle yq value for v
            vslice = xr.open_mfdataset(
                f"/home/149/ab8992/bottom_near_inertial_waves/{expt_suite_name}/{self.exptname}/{self.runname}/archive/output00*/v_10min.nc",chunks = {"zi":1},decode_times = False).isel({"yq" : sliceint,"time" : slice(0,4152//2)})
            uslice = xr.open_mfdataset(
                f"/home/149/ab8992/bottom_near_inertial_waves/{expt_suite_name}/{self.exptname}/{self.runname}/archive/output00*/u_10min.nc",chunks = {"zi":1},decode_times = False).isel({otherdim : sliceint,"time" : slice(0,4152//2)}).interp(xq = xh).rename({"xq":"xh"})
            tauslice = xr.open_dataset(
                f"/home/149/ab8992/bottom_near_inertial_waves/{expt_suite_name}/{self.exptname}/{self.runname}/archive/output000/taux_10min.nc",decode_times = False).isel({otherdim : sliceint}).interp(xq = xh).rename({"xq":"xh"})

        if merid == True:
            outpath = basepath +  "/" + foldername + "/merid_" + self.runname ## Modification to allow function to work as class

            dim = "yh"
            otherdim = "xh"
            sliceint = 2
            eslice = xr.open_mfdataset(
                f"/home/149/ab8992/bottom_near_inertial_waves/{expt_suite_name}/{self.exptname}/{self.runname}/archive/output00*/e_10min.nc",chunks = {"zi":1,"time":500,dim:500},decode_times = False).isel({otherdim : sliceint,"time" : slice(0,4152//2)}).isel(zi = slice(1,None))

            vslice = xr.open_mfdataset(
                f"/home/149/ab8992/bottom_near_inertial_waves/{expt_suite_name}/{self.exptname}/{self.runname}/archive/output00*/v_10min.nc",chunks = {"zl":1,"time":500,dim:500},decode_times = False).isel({"xh" : sliceint,"time" : slice(0,4152//2)}).interp(yq=yh).rename({"yq":"yh"})
            uslice = xr.open_mfdataset(
                f"/home/149/ab8992/bottom_near_inertial_waves/{expt_suite_name}/{self.exptname}/{self.runname}/archive/output00*/u_10min.nc",chunks = {"zl":1,"time":500,dim:500},decode_times = False).isel({"xq" : sliceint,"time" : slice(0,4152//2)})
            tauslice = xr.open_mfdataset(
                f"/home/149/ab8992/bottom_near_inertial_waves/{expt_suite_name}/{self.exptname}/{self.runname}/archive/output000/taux_10min.nc",decode_times = False).isel({"xq" : sliceint})

        forcing_time = tauslice.time.shape[0]





        e = eslice.e.assign_coords({"time":("time",eslice.time.values * 60),
                                dim:(dim,eslice[dim].values * 1000)
            })
        u = uslice.u.assign_coords({"time":("time",uslice.time.values * 60),
                                dim:(dim,uslice[dim].values * 1000)
            })

        v = vslice.v.assign_coords({"time":("time",vslice.time.values * 60),
                                dim:(dim,vslice[dim].values * 1000)
            })

        tau = tauslice.taux.assign_coords({"time":("time",tauslice.time.values * 60),
                                dim:(dim,tauslice[dim].values * 1000)
            })


        eanom = (e - e.mean(dim))
        eslice.close()
        uslice.close()
        vslice.close()
        tauslice.close()
        print("files closed")
        ## Flip and double time axis to remove ringing from FFT
        eanom_flipped = xr.DataArray(
            data = eanom[::-1,:,:],
            coords = [600 - eanom.time[::-1],eanom.zi,eanom[dim]],
            dims = ["time","zi",dim]
        )
        u_flipped = xr.DataArray(
            data = u[::-1,:,:],
            coords = [600 - u.time[::-1],u.zl,u[dim]],
            dims = ["time","zl",dim]
        )
        v_flipped = xr.DataArray(
            data = v[::-1,:,:],
            coords = [600 - v.time[::-1],v.zl,v[dim]],
            dims = ["time","zl",dim]
        )
        print("load e")
        e_doubled = xr.concat([eanom_flipped,eanom],dim = "time").load().chunk({"time":-1,dim:-1,"zi":1})
        print("load u")
        u_doubled = xr.concat([u_flipped,u],dim = "time").load().chunk({"time":-1,dim:-1,"zl":1})
        print("load v")
        v_doubled = xr.concat([v_flipped,v],dim = "time").load().chunk({"time":-1,dim:-1,"zl":1})
        print("Rechunked")
        del u_flipped
        del v_flipped
        del eanom_flipped

        strat = 1
        if "strat" in self.runname:
            strat = float(self.runname.split("-")[-1])

        drho = (e.zi[1] - e.zi[0]).values

        gprime = strat * 9.8 * drho / 1027
        f = 0.0001 / (2 * np.pi)
        print("Calculate PE")
        PE = 1027 * gprime * xrft.power_spectrum(e_doubled,dim=["time",dim],true_phase = True,true_amplitude = True)
        print("Calculate KE")
        # Scale KE by layer thickness, density and divide by 2 from 0.5v^2
        KE = (4000 / 20) * 1027 * 0.5 * (
            xrft.power_spectrum(u_doubled,dim=["time",dim],true_phase = True,true_amplitude = True)
            + xrft.power_spectrum(v_doubled,dim=["time",dim],true_phase = True,true_amplitude = True))

        ## ADD THIS LINE TO REMOVE SUBINERTIAL SIGNALS
        # PE_filtered = PE.where(np.abs(PE.freq_time) > f,0)
        # KE_filtered = KE.where(np.abs(KE.freq_time) > f,0).where(KE.freq_xh != 0,0)
        PEtot =  PE.where(np.abs(PE.freq_time) > f,0).sum("freq_time").sum(f"freq_{dim}") / (e_doubled.time.shape[0] * 600)
        KEtot =  KE.where(np.abs(KE.freq_time) > f,0).where(KE[f"freq_{dim}"] != 0,0).sum("freq_time").sum(f"freq_{dim}") / (e_doubled.time.shape[0] * 600)
        print("Calculate WW")
        ww = (u.isel(zl = 0,time = slice(0,forcing_time)) * tau)
        ww_tot = ww.integrate("time").integrate(dim)

        ## Commentted out include spectrum in saved files
        # out = xr.Dataset(
        #     {"PE":PE.sel({"freq_time" : slice(0,None),f"freq_{dim}" : slice(0,None)}),
        #     "PEtot":PEtot,
        #     "KEtot":KEtot,
        #     "KE":KE.sel({"freq_time" : slice(0,None),f"freq_{dim}" : slice(0,None)}),
        #     "ww_tot":ww_tot}
        # )
        print("Saving..")
        out = xr.Dataset(
            {
            "PEtot":PEtot,
            "KEtot":KEtot,
            "ww_tot":ww_tot}
        )

        out.to_netcdf(outpath)






### COMMON ACROSS ALL EXPERIMENTS. 
##########################################################
nlayers = 20             # number of layers
Lx = 4000                # domain zonal extent [km]
Ly = 1000                # domain meridional extent [km]
H  = 4000                # total fluid's depth in [m]
gridspacing = 2          # in [km]
flat = False
reverse = True
x = np.arange(0, Lx, gridspacing)
y = np.arange(-Ly/2, Ly/2, gridspacing)
ninterf = nlayers + 1    # number of interfaces
interfaces = np.arange(0,  ninterf)
nx = int(round(Lx/gridspacing))
ny = int(round(Ly/gridspacing))
overrides = ["ADIABATIC = True",
             "RHO_0 = 1027.0",
             f"NJGLOBAL={ny}",
             f"NIGLOBAL={nx}",
             f"LENLON = {Lx}.0",
             f"WESTLON = -{Lx//2}.0",
             f"LENLAT = {Ly}.0",
             f"WESTLAT = -{Ly//2}.0"]
default_dir = "default_rundir"


## Default values when unperturbed
default_height = 500 # m
default_forcing_latwidth = 300 # km
default_duration = 5 # hours
default_layerdensities = np.linspace(1027,1029,nlayers) ## This is the default stratification
default_strength = 1
default_ridge_width = 12.5 # km

## Perterbation values
## strat variables 
strats = [0.25,0.2,0.5,0.75,1,1.25,1.5,2,3,4,0.001,0.01,6]

## Height
heights = [10,20,50,80,120,150,225,275,350,450,600,750,1000,1250,1500,2000] + [500,650,700]

## Duration
durations = [0.5,1,2,3,4,5,6,8.7,10] + [7,8,12,14,15,16,20] ## Note! Will need to modify the amount of time that model runs for with forcing in this case

## Strength
strengths = [0.1,0.2,0.5,0.75,1,1.25,1.5,2,3,4,10] + [5,6,7,8,9]

## Topog Width
widths = [4,8,12.5,16,25,35,50,100,200] + [75,125,150,300,400,500,600,700,800,1000]

## Forcing Width
forcing_widths = [25,50,75,100,150,175,200,300,500,800,1000] + [250,400]



expts = []
k = [20]
topos = ["ridge"]


for nlayers in k:
    for topo in topos:
        # ## HEIGHTS
        # for height in heights:
        #     run = expt(x,y,nlayers,"height",height,topo,"windforcing")
        #     expts.append(run)

        # ## STRATS
        # for strat in strats:
        #     run = expt(x,y,nlayers,"strat",strat,topo,"windforcing")
        #     expts.append(run)

        # ## TOPO WIDTHS
        # for width in widths:
        #     run = expt(x,y,nlayers,"width",width,topo,"windforcing")
        #     expts.append(run)


        ## WIND FORCING EXPERIMENTS
        # Here we recycle the wind forcing since it doesn't change with topo
        ## FORCING WIDTHS

        if expt == "forcing_latwidth":

            for forcing_width in forcing_widths:
                run = expt(x,y,nlayers,"forcing_latwidth",forcing_width,topo,"topog")
                expts.append(run)

        if expt == "strength":
        ## STRENGTHS
            for strength in strengths:
                run = expt(x,y,nlayers,"strength",strength,topo,"topog")
                expts.append(run)

        if expt == "duration":
                ## DURATIONS
            for duration in durations:
                run = expt(x,y,nlayers,"duration",duration,topo,"topog")
                expts.append(run)


from IPython.display import clear_output


tot = len(expts)
curr = 0
for i in expts:
    print(f"{curr}/{tot} {100 * curr/tot}% complete. \t\t Processing {i.runname}. ")
    i.process_output(foldername = expt_suite_name , merid = True)
    i.process_output(foldername = expt_suite_name , merid = False)

    curr += 1
# up to 45 i.e strength 1