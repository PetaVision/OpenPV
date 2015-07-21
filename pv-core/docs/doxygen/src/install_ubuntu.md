Ubuntu Installation
==================================

This is a tutorial on getting PetaVision and running on Ubuntu-14.04 with GPU support, assuming you have a clean installation of Ubuntu.

Requirements
----------------------------------
- An NVIDIA card with compute capabilities 3.0 or above. Check at <https://developer.nvidia.com/cuda-gpus> to see if your video card is supported
- Ubuntu 14.04
- sudo access


Initialization
----------------------------------
~~~~~~~~~~~~~~~~~~~~{.sh}
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install cmake cmake-curses-gui gcc g++ gdb subversion
~~~~~~~~~~~~~~~~~~~~


OpenMPI
----------------------------------
~~~~~~~~~~~~~~~~~~~~{.sh}
sudo apt-get install libopenmpi-dev openmpi-bin
~~~~~~~~~~~~~~~~~~~~


GDAL
----------------------------------
~~~~~~~~~~~~~~~~~~~~{.sh}
sudo apt-get install libgdal-dev gdal-bin
~~~~~~~~~~~~~~~~~~~~


CUDA and NVIDIA Driver
----------------------------------
Go to <http://www.nvidia.com/Download/index.aspx?lang=en-us> and find your video card. Download the linux RUN file. Note the location.
Go to <https://developer.nvidia.com/cuda-downloads>.
Click the RUN file for your archetecture. Make sure if you have a 64 bit version of ubuntu that you download the 64 bit version cuda. Note the location.

NVIDIA drivers may not be installed unless X is not running. Here's how to install the drivers.

Reboot Ubuntu. At the GRUB menu, select Acvanced options for Ubuntu.
Boot Ubuntu into recovery mode.
Select root to get into root command line.
~~~~~~~~~~~~~~~~~~~~{.sh}
mount -o remount,rw #This remounts your drive into read/write mode
cd path/to/cuda.run 
chmod u+x path/to/cuda/driver.run
./path/to/cuda/driver.ruh
~~~~~~~~~~~~~~~~~~~~

Follow the onscreen instructions.
When it asks if you want to install NVIDIA driver, select yes. This version is outdated, but cuda needs this driver to work.
Make sure to install CUDA 6.5 toolkit into the default location. Add a symbolic link to `/usr/local/cuda` when asked.

~~~~~~~~~~~~~~~~~~~~{.sh}
cd path/to/driver.run
chmod u+x path/to/driver.run
./path/to/driver.run
~~~~~~~~~~~~~~~~~~~~

Follow the instructions on the screen.
When asked to uninstall the driver installed by cuda, select yes.
When asked to update your Xorg, select yes. This will use this driver as your GUI display.
Run `nvidia-smi` and make sure you see your video card.
Reboot.

CUDNN
----------------------------------
Go to <https://developer.nvidia.com/cuDNN> and click Download at the bottom.
Register with NVIDIA developers if need be, and wait for confirmation.
Download the Linux version of CUDNN
(Optional: if you have access to NMC's compuneuro, grab it from here: `/nh/compneuro/Data/cuDNN`)

Finalization
----------------------------------
Do one final update/upgrade for everything
~~~~~~~~~~~~~~~~~~~~{.sh}
sudo apt-get update
sudo apt-get upgrade
~~~~~~~~~~~~~~~~~~~~


Checking Out and Installing PetaVision
----------------------------------

~~~~~~~~~~~~~~~~~~~~{.sh}
cd ${HOME}
mkdir workspace
cd workspace
svn co https://<useranme>@svn.code.sf.net/p/petavision/code/trunk PetaVision
svn co https://<username>@svn.code.sf.net/p/petavision/code/PVSystemTests PVSystemTests
#You can download your sandbox here
cp PetaVision/docs/cmake/CMakeLists.txt .
#Here, you would add your sandbox to the end of CMakeLists
ccmake .
~~~~~~~~~~~~~~~~~~~~

CMake options:
~~~~~~~~~~~~~~~~~~~~
CMAKE_BUILD_TYPE = Release #Optimiztaion for CPU
CUDA_GPU = True #Depending on if you want to use GPUS
CUDA_RELEASE = True #Optimization for Cuda
CUDNN = True #If you're using CUDNN
CUDNN_PATH = /path/to/cudnn/folder #The path to the cuDNN folder you downloaded/copied
OPEN_MP_THREADS = True #If we use threads or not
~~~~~~~~~~~~~~~~~~~~

If some of these options do not show up on ccmake, fill the ones you can, press c to configure, and look again for variables
Press G to generate when avaliable.

To Test:

~~~~~~~~~~~~~~~~~~~~{.sh}
cd ~/workspace/PVSystemTest/GPUSystemTest
make -j 8
Release/GPUSystemTest -p input/postTest.params -t
~~~~~~~~~~~~~~~~~~~~


Optional: Python
----------------------------------
Ubuntu comes with python 2.7. We suggest installing anaconda python for analysis tools.
- Go to <https://store.continuum.io/cshop/anaconda/>
- Enter your email
- Click I WANT PYTHON 3.4
- Click the linux penguin
- Click the download link and save it

~~~~~~~~~~~~~~~~~~~~{.sh}
chmod u+x path/to/python.sh
./path/to/python.sh
~~~~~~~~~~~~~~~~~~~~

Follow the directions. Make sure to specify yes when it asks you if you want anaconda in your path.

~~~~~~~~~~~~~~~~~~~~{.sh}
source ~/.bashrc
~~~~~~~~~~~~~~~~~~~~

To ensure it's working, type python and make sure the text splash says `Python 3.4.1 |Anaconda 2.1`


Optional: Octave
----------------------------------

~~~~~~~~~~~~~~~~~~~~{.sh}
sudo apt-get install octave liboctave-dev
octave
~~~~~~~~~~~~~~~~~~~~

In octave:
~~~~~~~~~~~~~~~~~~~~{.m}
pkg install -forge general
pkg install -forge control
pkg install -forge signal
pkg install -forge image
pkg install -forge parallel
exit
~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~{.sh}
echo pkg load all >> ~/.octaverc
~~~~~~~~~~~~~~~~~~~~
