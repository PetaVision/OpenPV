OS X Installation
==================================

This is a tutorial on getting PetaVision installed and running on OS X, assuming you have a clean installation of OS X. You will
need an admin account (for installing packages with sudo). Tested on OS X 10.10.5 (Yosemite). If you have an NVIDIA card and
upgrade to Yosemite, you will need to reinstall cuDNN.

Requirements
----------------------------------
- OS X
- sudo access

Required installations
----------------------------------
- Xcode and Xcode command line tools
- Homebrew
- GDAL

Optional installations
----------------------------------
- OpenMPI (to take advantage of MPI parallelization)
- CUDA and cuDNN (to take advantage of CUDA GPU acceleration)
- A C/C++ compiler with OpenMP capabilities (to take advantage of OpenMP parallelization)
- Octave (to use the m-files in the mlab directory for analysis)

Xcode
----------------------------------
Xcode is needed to install Homebrew and C/C++ compilers. Here's how to get it.
- Go to the Apple App Store and search for Xcode
- Click 'GET', then 'INSTALL'. Put in your Apple ID information
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
Homebrew is a package manager for OS X, which we will use for many of PetaVision's dependencies. Additional information can be found at <http://brew.sh>
To install homebrew:

~~~~~~~~~~~~~~~~~~~~~~~~~{.sh}
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
~~~~~~~~~~~~~~~~~~~~~~~~~

Initialization
----------------------------------

~~~~~~~~~~~~~~~~~~~~~~~~~{.sh}
brew update
brew install cmake
brew install wget
brew install gdal
# optionally, to install OpenMPI:
brew install open-mpi
# optionally, to install Octave using homebrew
# (note: installation of Octave on a fresh system typically takes several hours):
brew tap homebrew/science
brew install octave
~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to take advantage of OpenMP threading, see below.

OpenMP-compatible Clang (optional)
----------------------------------
Currently (OS X Yosemite 10.10.5 and Xcode 6.4), the version of Clang installed by Xcode does not support OpenMP.
The program installed as gcc is also Clang, not GNU-GCC (as can be verified by running gcc --version).
To make use of the OpenMP threading in PetaVision you will need an OpenMP-compatible compiler. 
As of this writing, the most recent release of Clang (3.7.0) supports OpenMP, but Xcode is still behind
this version.
Here are instructions to download the most recent version of clang.

###Install the OMP-compatible version of Clang ###############
Download the most recent pre-built binary for Mac OS X from <http://llvm.org/releases/download.html>.
Make a note of the location of the downloaded file and the filename.  In what follows, we will
be using the filename `clang+llvm-3.7.0-x86_64-apple-darwin.tar.xz`, the most recent version as of
this writing; modify as necessary.  To extract the downloaded `.xz` file using the command line,
you will need the xz program, which can be installed using Homebrew.

~~~~~~~~~~~~~~~~~~~~~~~~~{.sh}
brew install xz
cd ${HOME}
mkdir clang_omp
cd clang_omp
cp /path/to/clang+llvm-3.7.0-x86_64-apple-darwin.tar.xz .
unxz clang+llvm-3.7.0-x86_64-apple-darwin.tar.xz
tar xf clang+llvm-3.7.0-x86_64-apple-darwin.tar
cd clang+llvm-3.7.0-x86_64-apple-darwin/bin
ln -s clang cc
ln -s clang++ c++
~~~~~~~~~~~~~~~~~~~~~~~~~

###Install the OMP library ###############
Download the most recent version of the Intel OpenMP Runtime Library from <https://www.openmprtl.org/download#stable-releases>.
Make a note of the location of the downloaded file.  In what follows, we will be using `libomp_20150701_oss.tgz`,
the most recent version as of this writing; modify as necessary.
~~~~~~~~~~~~~~~~~~~~~~~~~{.sh}
cd ${HOME}/clang_omp
tar xf /path/to/libomp_20150701_oss.tgz
cd libomp_oss
cmake .
make
~~~~~~~~~~~~~~~~~~~~~~~~~


###Update necessary environmental variables ###############
Open your `~/.bash_profile` (or `~/.profile`, whichever one you use) and append these lines to the end of the file:
~~~~~~~~~~~~~~~~~~~~~~~~~{.sh}
export PATH=$HOME/clang_omp/clang+llvm-3.7.0-x86_64-apple-darwin/bin:$PATH

export C_INCLUDE_PATH=$HOME/clang_omp/clang+llvm-3.7.0-x86_64-apple-darwin/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$HOME/clang_omp/clang+llvm-3.7.0-x86_64-apple-darwin/include:$CPLUS_INCLUDE_PATH
export LIBRARY_PATH=$HOME/clang_omp/clang+llvm-3.7.0-x86_64-apple-darwin/lib:$LIBRARY_PATH
export DYLD_LIBRARY_PATH=$HOME/clang_omp/clang+llvm-3.7.0-x86_64-apple-darwin/lib:$DYLD_LIBRARY_PATH

export C_INCLUDE_PATH=$HOME/clang_omp/libomp_oss/exports/common/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$HOME/clang_omp/libomp_oss/exports/common/include:$CPLUS_INCLUDE_PATH
export LIBRARY_PATH=$HOME/clang_omp/libomp_oss/exports/mac_32e/lib.thin:$LIBRARY_PATH
export DYLD_LIBRARY_PATH=$HOME/clang_omp/libomp_oss/exports/mac_32e/lib.thin:$DYLD_LIBRARY_PATH
~~~~~~~~~~~~~~~~~~~~~~~~~

If you have any open terminal windows, run `source ~/.bash_profile` in each of them.
Test by running `which clang` and `which clang++` to verify that clang and clang++ resolve
to the versions just installed.

When you run cmake or ccmake, you will need to set PV_OPENMP_COMPILER_FLAG to "-fopenmp=libiomp5" in order for
the compiler to find the OpenMP Runtime library.

CUDA and NVIDIA Driver (optional):
----------------------------------
To take advantage of CUDA/cuDNN GPU acceleration your Macintosh needs to have an NVIDIA card with compute capabilities 3.0 or above. To find which video card you have, go to "About This Mac" under the Apple menu, select "System Report..." and then Graphics/Displays in the Hardware section.  Check at <https://developer.nvidia.com/cuda-gpus> to see if your video card supports CUDA.
NVIDIA drivers are included with the cuda download. To install cuda:
- Go to <https://developer.nvidia.com/cuda-downloads>
- Select Mac OS X, and then your version of OS X and download the local installer .dmg
- If the .dmg does not automatically open, double-click the .dmg file to mount the CUDAMacOSXInstaller disk.
- Open the CUDAMacOSXInstaller/CUDAMacOSXInstaller app on that disk and follow the onscreen instructions.
- Open the downloaded .pkg file and follow the onscreen instructions.
  On the Select Packages To Install page, make sure to select cuda driver and cuda toolkit.


cuDNN (optional)
----------------------------------
Go to <https://developer.nvidia.com/cuDNN> and click Download at the bottom.
Register with NVIDIA developers if need be, and wait for confirmation.
Download the OS X version of cuDNN
(Optional: if you have access to NMC's compuneuro, grab it from here: `/nh/compneuro/Data/cuDNN`)


Checking Out and Installing PetaVision:
----------------------------------

~~~~~~~~~~~~~~~~~~~~{.sh}
cd ${HOME}
git clone https://github.com/PetaVision/OpenPV
cd OpenPV
git clone https://github.com/PetaVision/OpenPV
cd OpenPV
echo PVSystemTests > subdirectories.txt
#Edit the file subdirectories.txt to include any desired projects or
#auxiliary libraries
ccmake .
~~~~~~~~~~~~~~~~~~~~

CMake options:
~~~~~~~~~~~~~~~~~~~~
CLANG_OMP = (ON or OFF) #Turn this on if using a version of Clang that supports OpenMP as opposed to the Apple-supplied clang.
CMAKE_BUILD_TYPE = (Release or Debug) #Whether to optimize for performance or for debugging
...
CUDA_RELEASE = (ON or OFF) #Whether to Optimization for Cuda
CUDNN_PATH = /path/to/cudnn/folder #The path to the cuDNN folder you downloaded/copied
...
PV_USE_CUDA = (ON or OFF) #Whether to use CUDA GPU acceleration
PV_USE_CUDNN = True #Whether to use cuDNN
PV_USE_OPENMP_THREADS = (ON or OFF) #Whether to use OpenMP threading
~~~~~~~~~~~~~~~~~~~~

Not all options will show up immediately the first time you run ccmake.  Fill the ones you can, press c to configure, and
look for new variables (marked with an asterisk).  Fill in those variables and repeat.

Note: if you are using OpenMP but not using OpenMPI, then you should also set the cmake-built-in variables CMAKE_C_COMPILER and CMAKE_CXX_COMPILER.  Press t to toggle advanced mode, and find the compiler variables.  Replace their values with the full path to the clang and clang++ that you installed earlier.  (If you are using OpenMPI, this is not necessary.  Turning PV_USE_MPI on will use the MPI-supplied compilers, which wrap around c/clang and c++/clang++.  Editing and exporting the PATH variable in .bash_profile or .profile made sure that the MPI compilers would find the new versions of the compilers.)

When all options have been set, press g to generate.  (If that option does not appear, there are either new variables -- press c to configure when they or filled in -- or there is an error in some of the existing values.)

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

