import numpy as np
import math as m
from scipy.constants import c
import h5py

#custom libraries
from Reconal import ray_utils, defs
from rayReader import *


##CONFIG

##Geometry
#> adjust parameters to reduce generation time/file sizes
#default values will take ~3.3hr, & create .h5 file >1GB

r_step = 0.01 #dr of traced ray

#Range of launch angles, measured from +z axis
theta_min = 90
theta_max = 180
d_theta = 0.01 #angular separation between rays

#grid over which ray is generated
r_range = rmin, rmax = 0,100
z_range = zmin, zmax = -100,0


#>keep this fixed
tx = [0,0] #fixed point on surface of ice, keep at origin


#Ice
ior = defs.ior_exp3
grad_ior = defs.grad_ior_exp3

#file containing generated rays, sorted by launch angle
outfile_name = 'ray_dictionary.h5'

#===================================================================#

def getLaunchAngle(arr_r,arr_z):
    length_r = np.round(arr_r[1]-arr_r[0],10)
    length_z = np.round(arr_z[1]-arr_z[0],10)
    theta_offset = 0
    if length_z<=0:
        theta_offset = 90
    launch_angle = np.rad2deg(np.arctan(np.abs(length_z/length_r))) + theta_offset
    return launch_angle


def generate_ray(launch_angle_deg, source_location, r_step=0.01, d_theta=0.01):
    theta_min,theta_max = launch_angle_deg - d_theta, launch_angle_deg + d_theta
    ray_mesh = np.linspace(theta_min, theta_max, 5)

    #Generating Rays
    rays, turnover = ray_utils.get_rays(source_location, ior, grad_ior, rmax, zmin, zmax, ray_mesh, r_step)

    precision = int(-m.log10(d_theta))
    launch_angles = np.array([np.round(getLaunchAngle(ray[0], ray[1]), precision) for ray in rays])


    sri = np.argmin(np.abs(launch_angles-launch_angle_deg))

    return rays[sri]


#======================================


scale = int(round(1 / d_theta))

# exact grid in integer space
i_vals = np.arange(theta_min * scale, theta_max * scale + 1, dtype=np.int32)
launch_angle_set = i_vals / scale

#keys for selecting rays, saved in degrees for readability
angle_labels = np.array([f"{i/scale:.2f}" for i in i_vals], dtype=h5py.string_dtype(encoding="utf-8"))
N_rays = len(launch_angle_set)


#create file and fill with rays, catch potential invalid rays (common near end cases)
with h5py.File(outfile_name, "w") as ray_dictionary:

    dset = ray_dictionary.create_dataset(
        "rays",
        shape=(len(launch_angle_set), 3, 100000),
        dtype=np.float32,
        chunks=(1, 3, 100000),
        compression="lzf"
    )

    valid_mask = np.zeros(len(launch_angle_set), dtype=np.bool_)

    for i, angle in enumerate(launch_angle_set):

        try:
            ray = generate_ray(
                angle,
                source_location=tx,
                r_step=r_step,
                d_theta=d_theta
            )
            print(f'Ray ({angle})[{angle_labels[i]}] finished')

            dset[i] = ray.astype(np.float32)
            valid_mask[i] = True

        except Exception as e:
            print(f"FAILED for ray {i} of {N_rays} (angle {angle:.2f}): {e}")

        if (i + 1) % 100 == 0:
            print(f"{i+1}/{len(launch_angle_set)} rays generated")

    #
    ray_dictionary.create_dataset("launch_angle_labels",data=angle_labels) #keys for selecting each ray,in degrees (e.g. "102.35")
    ray_dictionary.create_dataset("valid_mask",data=valid_mask) #mask to rule out any launch angles with invalid rays.

