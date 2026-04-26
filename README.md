# Ray Reader
Python class for reading traced rays from a .h5 file. Includes scripts to generate file with rays.

## Requirements

This generates rays using the numerical raytracer from Reconal
* https://github.com/philippwindischhofer/Reconal

## Scripts 

* ```rayReader.py```: Ray class, imported into python scripts, contains functions for loading rays & calculating quantities
* ```generate_rayset.ipynb``` : generate .h5 file containg rays (python notebook)
* ```generate_rayDict.py```: generate .h5 file containg rays (python file)
* ```testRayRead.ipynb```: Contains a couple examples of how to load and plot rays from .h5 file. 


# How to use w/ NuRadio (or from anywhere):

### Add ray reader to $PYTHONPATH  in your ```.bashrc/.zsh_rc```
> ```export PYTHONPATH="~/path/to/ray_reader/:$PYTHONPATH"```

### Add environment variable to your ```.bashrc/.zsh_rc``` after you generate .h5 file with rays.
> ```export RNO_G_RAYS_H5_FILE="~/path/to/ray_reader/ray_dictionary.h5" ```
 

## Generating file with rays:

You can choose one of two scripts to generate rays depending on if you prefer notebooks or command line ```generate_rayset.ipynb``` or ```generate_rayDict.py```. The code in both is identical.

IMPORTANT: The default resolution and range of angles will take a long time to generate a file, and it will be large. I'd recommend testing a smaller range of angles, smaller angle separation (```d_theta```), and maybe a smaller dr (```r_step```) for the ray in order to speed things up. When you have your preferred settings:

* ```python3 generate_rayDict.py```
* Or you can walk though the notebook in VSCode if you want to see the process in more detail.

# Geometry:

* Rays stored in the .h5 file are sorted by their launch angles from a point on the surface. 

* Each ray contains 3 arrays (r,z,t_propagation), and are padded with nans to ensure all arrays are the same size between ray paths.

* The launch angle is just the angle from the +Z axis at which the ray starts to travel beneath the ice's surface. It is the supplemental angle to the refracted zenith angle of the source (which is measured from -Z, after snell's law).

The rayReader class also contains methods for calculating the arrival time delays between two positions, assuming plane wave emission (one ray shape for all positions).





* 
