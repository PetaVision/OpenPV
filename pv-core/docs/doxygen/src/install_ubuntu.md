Ubuntu Installation
==================================

This page provides instructions for installing PetaVision on
Ubuntu&nbsp;14.04 (Trusty Tahr), beginning with a clean installation of Ubuntu.
You will need administrator privileges to run sudo.

Required packages
----------------------------------
~~~~~~~~~~~~~~~~~~~~{.sh}
sudo apt-get update
sudo apt-get dist-upgrade
sudo apt-get install bison cmake cmake-curses-gui flex gcc g++ git openmpi-bin libgdal-dev libopenmpi-dev openmpi-bin
~~~~~~~~~~~~~~~~~~~~

Optional packages
----------------------------------
~~~~~~~~~~~~~~~~~~~~{.sh}
sudo apt-get install lua5.2 # for using lua scripts to manage parameter files
sudo apt-get install octave octave-image # for using octave .m scripts in the analysis folder
echo pkg load all >> ~/.octaverc # loads octave image package etc. when starting octave.
sudo apt-get install gdb # Line debugger for gcc and g++
~~~~~~~~~~~~~~~~~~~~

CUDA and NVIDIA Driver (Optional)
----------------------------------
Our GPU acceleration depends on CUDA.  While it is possible to build and run
PetaVision without CUDA, performance will not be nearly as good.

To download CUDA, go to <https://developer.nvidia.com/cuda-downloads>.
Click the RUN file for your architecture and your version of Ubuntu. Note the location.

The version of the NVIDIA driver installed by CUDA may be behind the most
recent driver version released by NVIDIA.  If you want to use a more recent
driver, you will need to install CUDA first, along with the driver that CUDA
installs, and then upgrade the NVIDIA driver.  
If you want to use the most recent NVIDIA driver, go to
<http://www.nvidia.com/Download/index.aspx?lang=en-us> and select your
video card. Download the linux RUN file. Note the location.

NVIDIA drivers cannot be installed while the X window system is running. Here's how to install the drivers.

Enter a virtual console (usually by pressing control-alt-F1 or control-alt-fn-F1).
Log into an account with admin privileges.
~~~~~~~~~~~~~~~~~~~~{.sh}
sudo service lightdm stop
chmod u+x path/to/cuda.run
sudo path/to/cuda.run
~~~~~~~~~~~~~~~~~~~~

Follow the onscreen instructions.  We have seen the warning "The distribution-provided pre-install script failed!" but
it does not seem to cause problems if you continue.

When it asks if you want to install NVIDIA driver, select yes.  Add a symbolic link to `/usr/local/cuda` when asked.


If you now want to upgrade the NVIDIA driver to a version downloaded above.
~~~~~~~~~~~~~~~~~~~~{.sh}
cd path/to/driver.run
chmod u+x path/to/driver.run
./path/to/driver.run
~~~~~~~~~~~~~~~~~~~~

Follow the instructions on the screen.
When asked to uninstall the existing driver (installed by cuda), select yes.
When asked to update your Xorg, select yes. This will use this driver as your GUI display.

Run `nvidia-smi` and make sure you see your video card.

Reboot.

cuDNN (optional)
----------------------------------
You can optionally use the cuDNN Deep Neural Network Library, which provides additional
speed-up in computing the convolutions we use to compute input to post-synaptic layers.

To download cuDNN, go to <https://developer.nvidia.com/cuDNN> and click Download.
If you haven't already registered with NVIDIA Developer you will need to do so, and then wait for email confirmation.
Once registered, log into NVIDIA Developer and download the cuDNN Library for Linux.
(Alternatively, if you have access to NMC's compneuro, grab it from `/nh/compneuro/Data/cuDNN`)

Finalization
----------------------------------
Do one final update/upgrade for everything
~~~~~~~~~~~~~~~~~~~~{.sh}
sudo apt-get update
sudo apt-get dist-upgrade
~~~~~~~~~~~~~~~~~~~~


Checking Out and Installing PetaVision
----------------------------------

~~~~~~~~~~~~~~~~~~~~{.sh}
cd ${HOME}
mkdir workspace
cd workspace
git clone https://github.com/PetaVision/OpenPV
ccmake .
~~~~~~~~~~~~~~~~~~~~

CMake options:
~~~~~~~~~~~~~~~~~~~~
CMAKE_BUILD_TYPE = Release #Optimiztaion for CPU
PV_USE_CUDA = ON #Depending on if you want to use GPUS
CUDA_RELEASE = True #Optimization for Cuda
CUDNN = True #If you're using CUDNN
CUDNN_PATH = /path/to/cudnn/folder #The path to the cuDNN folder you downloaded/copied
PV_USE_OPENMP_THREADS = ON #Whether to use OpenMP threads for parallelization
~~~~~~~~~~~~~~~~~~~~

If some of these options do not show up on ccmake, fill the ones you can, press c to configure, and look again for variables
Press G to generate when avaliable.

Running the system tests:

~~~~~~~~~~~~~~~~~~~~{.sh}
cd ~/workspace/PVSystemTest
make
ctest
~~~~~~~~~~~~~~~~~~~~


Optional: Anaconda Python
----------------------------------
Ubuntu comes with python 2.7. We suggest installing Anaconda Python for analysis tools.
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
