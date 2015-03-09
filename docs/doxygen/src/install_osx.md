OSX Installation
==================================

This is a tutorial on getting PetaVision and running on OSX with GPU support, assuming you have a clean installation of OSX. Tested on OSX 10.9.5 (Mavericks). If you have an NVIDIA card, upgrading to Yosemite currently creates issues with cuDNN that are fixed by reinstalling cuDNN.


Requirements
----------------------------------
- An NVIDIA card with compute capabilities 3.0 or above. Check at <https://developer.nvidia.com/cuda-gpus> to see if your video card is supported
- OSX
- sudo access


XCode
----------------------------------
XCode is needed by homebrew and gcc/g++/clang. Here's how to get it.
- Go to the apple store and search for xcode
- Click get, then install. Put in your Apple account information
- After it's installed, we need to accept the xcode license.
   + Run the xcode application
   + Follow the onscreen prompt until you see the main xcode screen
   + Exit xcode
- We also need command line tools for xcode
   + Go into a terminal and execute the following command

~~~~~~~~~~~~~~~~~~~~~~~~~{.sh}
xcode-select --install
~~~~~~~~~~~~~~~~~~~~~~~~~

   + Follow the onscreen prompt to install command line tools.


Homebrew
----------------------------------
Homebrew is a package manager for OSX, which we will use to install all of PetaVision's dependencies. Additional information can be found at <http://brew.sh>
To install homebrew:

~~~~~~~~~~~~~~~~~~~~~~~~~{.sh}
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
~~~~~~~~~~~~~~~~~~~~~~~~~

Initialization
----------------------------------

~~~~~~~~~~~~~~~~~~~~~~~~~{.sh}
brew update
brew install svn
brew install cmake
brew install wget
~~~~~~~~~~~~~~~~~~~~~~~~~

Clang + OMP
----------------------------------
Currently, OSX's Clang does not support openMP. Therefore, we need to download a different version of clang and replace the current clang.

###Replace OSX default clang with clang+omp ###############

~~~~~~~~~~~~~~~~~~~~~~~~~{.sh}
cd ${HOME}
mkdir clamp
cd clamp
git clone https://github.com/clang-omp/llvm
git clone https://github.com/clang-omp/compiler-rt llvm/projects/compiler-rt
git clone -b clang-omp https://github.com/clang-omp/clang llvm/tools/clang
mkdir build
cd build
../llvm/configure --enable-optimized
make -j8
~~~~~~~~~~~~~~~~~~~~~~~~~

Open your `~/.bash_profile` (or `~/.profile`, whichever one you use) and append these lines to the end of the file, making sure the clamp path is before `/usr/bin`

~~~~~~~~~~~~~~~~~~~~~~~~~{.sh}
export PATH=~/clamp/build/Release+Asserts/bin:$PATH 
export C_INCLUDE_PATH=~/clamp/build/Release+Asserts/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=~/clamp/build/Release+Asserts/include:$CPLUS_INCLUDE_PATH
export LIBRARY_PATH=~/clamp/build/Release+Asserts/lib:$LIBRARY_PATH
export DYLD_LIBRARY_PATH=~/clamp/build/Release+Asserts/lib:$DYLD_LIBRARY_PATH
~~~~~~~~~~~~~~~~~~~~~~~~~

Make sure to run `source ~/.bash_profile`.
Run `which clang` to make sure it's pointing to `${HOME}/clamp/build/Release_Asserts/bin/clang`.

###Install the openmp runtime library ##################
Releases can be found at <https://www.openmprtl.org/download#stable-releases>

~~~~~~~~~~~~~~~~~~~~~~~~~{.sh}
mv ~/Downloads/name_of_openmp_runtime_library.tgz ~/clamp
cd ~/clamp
tar -xvf name_of_openmp_runtime_library.tgz
cd libomp_oss
cmake CMakeLists.txt
make -j8
~~~~~~~~~~~~~~~~~~~~~~~~~

Once again, append these lines to the end of your `~/.bash_profile`.

~~~~~~~~~~~~~~~~~~~~~~~~~{.sh}
export C_INCLUDE_PATH=~/clamp/libomp_oss:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=~/clamp/libomp_oss:$CPLUS_INCLUDE_PATH
export LIBRARY_PATH=~/clamp/libomp_oss:$LIBRARY_PATH
export DYLD_LIBRARY_PATH=~/clamp/libomp_oss:$DYLD_LIBRARY_PATH
~~~~~~~~~~~~~~~~~~~~~~~~~

Make sure to run `source ~/.bash_profile`.


OpenMPI
----------------------------------
With the new version of clang installed, we can now install openmpi using the new clang.

~~~~~~~~~~~~~~~~~~~~~~~~~{.sh}
brew install openmpi
~~~~~~~~~~~~~~~~~~~~~~~~~


GDAL
----------------------------------
~~~~~~~~~~~~~~~~~~~~~~~~~{.sh}
brew install gdal
~~~~~~~~~~~~~~~~~~~~~~~~~


CUDA and NVIDIA Driver:
----------------------------------
NVIDIA drivers are included with the cuda download. To install cuda:
- Go to <https://developer.nvidia.com/cuda-downloads>
- Select Mac OSX and download the pkg provided
- Follow the onscreen instructions. Make sure to select cuda driver and cuda toolkit.


CUDNN
----------------------------------
Go to <https://developer.nvidia.com/cuDNN> and click Download at the bottom.
Register with NVIDIA developers if need be, and wait for confirmation.
Download the OSX version of CUDNN
(Optional: if you have access to NMC's compuneuro, grab it from here: `/nh/compneuro/Data/cuDNN`)


Checking Out and Installing PetaVision:
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
CLANG_OMP = True #Tells cmake to use new clang
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

