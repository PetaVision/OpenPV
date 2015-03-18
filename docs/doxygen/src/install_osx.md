OS X Installation
==================================

This is a tutorial on getting PetaVision installed and running on OS X, assuming you have a clean installation of OS X. You will need an admin account (for installing packages with sudo). Tested on OS X 10.9.5 (Mavericks). If you have an NVIDIA card, upgrading to Yosemite currently creates issues with cuDNN that are fixed by reinstalling cuDNN.

Requirements
----------------------------------
- OS X
- sudo access

Required installations
----------------------------------
- Xcode and Xcode command line tools
- Homebrew
- GDAL
- OpenMPI

Optional installations
----------------------------------
- CUDA and cuDNN (to take advantage of CUDA GPU acceleration)
- A C/C++ compiler with OpenMP capabilities (to take advantage of OpenMP parallelization)
- Octave (to use the m-files in the mlab directory for analysis)

Xcode
----------------------------------
Xcode is needed by homebrew and gcc/g++/clang. Here's how to get it.
- Go to the Apple App Store and search for Xcode
- Click get, then install. Put in your Apple ID information
- After it's installed, we need to accept the Xcode license.
   + Run the Xcode application
   + Follow the onscreen prompt until you see the main Xcode screen
   + Exit Xcode
- We also need command line tools for Xcode
   + Go into a terminal and execute the following command

~~~~~~~~~~~~~~~~~~~~~~~~~{.sh}
xcode-select --install
~~~~~~~~~~~~~~~~~~~~~~~~~

   + Follow the onscreen prompt to install command line tools.


Homebrew
----------------------------------
Homebrew is a package manager for OS X, which we will use to install all of PetaVision's dependencies. Additional information can be found at <http://brew.sh>
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
# optionally, to install Octave using homebrew:
brew tap homebrew/science
brew install octave
~~~~~~~~~~~~~~~~~~~~~~~~~

Clang + OMP (optional)
----------------------------------
Currently (OS X Yosemite), the version of Clang installed by Xcode does not support OpenMP.  The program installed as gcc is also Clang, not GNU-GCC (as can be verified by running gcc --version). To make use of the OpenMP threading in PetaVision you will need an OpenMP compatible compiler.  Here are instructions to download a different version of clang and replace the current clang.

###Replace OS X default clang with clang+omp ###############

~~~~~~~~~~~~~~~~~~~~~~~~~{.sh}
cd ${HOME}
mkdir clang-omp
cd clang-omp
git clone https://github.com/clang-omp/llvm
git clone https://github.com/clang-omp/compiler-rt llvm/projects/compiler-rt
git clone -b clang-omp https://github.com/clang-omp/clang llvm/tools/clang
mkdir build
cd build
../llvm/configure --enable-optimized
make -j8
~~~~~~~~~~~~~~~~~~~~~~~~~

Open your `~/.bash_profile` (or `~/.profile`, whichever one you use) and append these lines to the end of the file, making sure the clang-omp path is before `/usr/bin`

~~~~~~~~~~~~~~~~~~~~~~~~~{.sh}
export PATH=~/clang-omp/build/Release+Asserts/bin:$PATH 
export C_INCLUDE_PATH=~/clang-omp/build/Release+Asserts/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=~/clang-omp/build/Release+Asserts/include:$CPLUS_INCLUDE_PATH
export LIBRARY_PATH=~/clang-omp/build/Release+Asserts/lib:$LIBRARY_PATH
export DYLD_LIBRARY_PATH=~/clang-omp/build/Release+Asserts/lib:$DYLD_LIBRARY_PATH
~~~~~~~~~~~~~~~~~~~~~~~~~

In any open terminals, run `source ~/.bash_profile`.
Run `which clang` to verify that it's pointing to `${HOME}/clang-omp/build/Release_Asserts/bin/clang`.

###Install the openmp runtime library ##################
Releases can be found at <https://www.openmprtl.org/download#stable-releases>

~~~~~~~~~~~~~~~~~~~~~~~~~{.sh}
mv ~/Downloads/name_of_openmp_runtime_library.tgz ~/clang-omp
cd ~/clang-omp
tar -xvf name_of_openmp_runtime_library.tgz
cd libomp_oss
cmake CMakeLists.txt
make -j8
~~~~~~~~~~~~~~~~~~~~~~~~~

Once again, append these lines to the end of your `~/.bash_profile`.

~~~~~~~~~~~~~~~~~~~~~~~~~{.sh}
export C_INCLUDE_PATH=~/clang-omp/libomp_oss:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=~/clang-omp/libomp_oss:$CPLUS_INCLUDE_PATH
export LIBRARY_PATH=~/clang-omp/libomp_oss:$LIBRARY_PATH
export DYLD_LIBRARY_PATH=~/clang-omp/libomp_oss:$DYLD_LIBRARY_PATH
~~~~~~~~~~~~~~~~~~~~~~~~~

Make sure to run `source ~/.bash_profile`.


OpenMPI
----------------------------------
With the new version of Clang installed, we can now install Open MPI using the new Clang.

~~~~~~~~~~~~~~~~~~~~~~~~~{.sh}
brew install openmpi
~~~~~~~~~~~~~~~~~~~~~~~~~


GDAL
----------------------------------
~~~~~~~~~~~~~~~~~~~~~~~~~{.sh}
brew install gdal
~~~~~~~~~~~~~~~~~~~~~~~~~


CUDA and NVIDIA Driver (optional):
----------------------------------
To take advantage of CUDA/cuDNN GPU acceleration your Macintosh needs to have an NVIDIA card with compute capabilities 3.0 or above. To find which video card you have, go to "About This Mac" under the Apple menu, select "System Report..." and then Graphics/Displays in the Hardware section.  Check at <https://developer.nvidia.com/cuda-gpus> to see if your video card supports CUDA.
NVIDIA drivers are included with the cuda download. To install cuda:
- Go to <https://developer.nvidia.com/cuda-downloads>
- Select Mac OS X and download the pkg provided
- Follow the onscreen instructions. Make sure to select cuda driver and cuda toolkit.


CUDNN (optional)
----------------------------------
Go to <https://developer.nvidia.com/cuDNN> and click Download at the bottom.
Register with NVIDIA developers if need be, and wait for confirmation.
Download the OS X version of CUDNN
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
#Here, you would add your sandbox to the end of CMakeLists.txt
ccmake .
~~~~~~~~~~~~~~~~~~~~

CMake options:
~~~~~~~~~~~~~~~~~~~~
CLANG_OMP = (ON or OFF) #Whether the compiler uses the OpenMP/Clang compiler
CMAKE_BUILD_TYPE = (Release or Debug) #Whether to optimize for performance or for debugging
CUDA_GPU = (ON or OFF) #Whether to use CUDA GPU acceleration
CUDA_RELEASE = (ON or OFF) #Whether to Optimization for Cuda
CUDNN = True #Whether to use CUDNN
CUDNN_PATH = /path/to/cudnn/folder #The path to the cuDNN folder you downloaded/copied
OPEN_MP_THREADS = (ON or OFF) #Whether to use OpenMP threading
~~~~~~~~~~~~~~~~~~~~

If some of these options do not show up on ccmake, fill the ones you can, press c to configure, and look again for variables
Press G to generate when avaliable.

To Test:
~~~~~~~~~~~~~~~~~~~~{.sh}
cd ~/workspace/PVSystemTest/BasicSystemTest
make -j 8
# Change Release to Debug in the line below if you built with CMAKE_BUILD_TYPE = Debug.
Release/BasicSystemTest -p input/BasicSystemTest.params -t
~~~~~~~~~~~~~~~~~~~~


To Test GPU capability:

~~~~~~~~~~~~~~~~~~~~{.sh}
cd ~/workspace/PVSystemTest/GPUSystemTest
make -j 8
# Change Release to Debug in the line below if you built with CMAKE_BUILD_TYPE = Debug.
Release/GPUSystemTest -p input/postTest.params -t
~~~~~~~~~~~~~~~~~~~~

