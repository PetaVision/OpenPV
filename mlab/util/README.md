# How to use #
Clone files to your local PetaVision folder (where your input, output and other folders are located). 

In general the scripts are started by *.sh files. They have the extension *.sh.template in the repository. Delete the extension part ".template" and change settings locally. These local files are ignored by git (.gitignore)

### errorcurve.sh
Plots the L1-norm (i.e. sum) of the activity, the L2-norm of the reconstruction error (i.e. vector length) and a weighted sum for both. Three windows are going to appear: one with the first n display periods, one with the last m display periods and one with an overall error curve for only the settled value just before the next display period (shows wether the error improves over all).

Settings can be adjusted for n and m. 
Attention: display period has to be set manually! Otherwise, the wrong overall curve will be plotted!

### weightsplot.sh
Creates folder ./weights_movie and creates one plots of weights per checkpoint. 
In the case of fish data, additionaly a vector fiel plot for each receptive field is created for the last checkpoint.

### sortweightsplot.sh
Like weightsplot.sh but sorted by average activity, gathered from the activity file of the HyperLCA-layer (V1.pvp, Pretectum.pvp)

## Additional information ##

Generally, these scripts should run both in Octave as well as in Matlab. 
These scripts will depend on other scripts shipped with PetaVision, so you 
have to add these to your path. In Octave that can be done by adding the 
following line to a `~/.octaverc` file. In Matlab the same line works inside a 
`startup.m`-file located anywhere in the search path:

```bash
addpath("~/OpenPV/mlab/util")
```
