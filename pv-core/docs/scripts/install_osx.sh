pause() {
  local dummy
  read -s -r -p "Press any key to continue..." -n 1 dummy
}

#xcode command line tools
xcode-select --install
pause


#Initialization
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
brew update
brew install svn
brew install git
brew install cmake
brew install wget
brew install gcc

#Clang + omp
cd ~
mkdir clamp
cd clamp
git clone https://github.com/clang-omp/llvm
git clone https://github.com/clang-omp/compiler-rt llvm/projects/compiler-rt
git clone -b clang-omp https://github.com/clang-omp/clang llvm/tools/clang
mkdir build
cd build
../llvm/configure --enable-optimized
make -j8

echo '
#New clang placed before clang that came with osx
export PATH=~/clamp/build/Release+Asserts/bin:$PATH 
export C_INCLUDE_PATH=~/clamp/build/Release+Asserts/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=~/clamp/build/Release+Asserts/include:$CPLUS_INCLUDE_PATH
export LIBRARY_PATH=~/clamp/build/Release+Asserts/lib:$LIBRARY_PATH
export DYLD_LIBRARY_PATH=~/clamp/build/Release+Asserts/lib:$DYLD_LIBRARY_PATH' >> ~/.bash_profile

source ~/.bash_profile

cd ~/clamp
wget https://www.openmprtl.org/sites/default/files/libomp_20141212_oss.tgz
tar -xvf libomp_20141212_oss.tgz
cd libomp_oss
cmake CMakeLists.txt
make -j8

echo '
#For omp runtime
export C_INCLUDE_PATH=~/clamp/libomp_oss:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=~/clamp/libomp_oss:$CPLUS_INCLUDE_PATH
export LIBRARY_PATH=~/clamp/libomp_oss:$LIBRARY_PATH
export DYLD_LIBRARY_PATH=~/clamp/libomp_oss:$DYLD_LIBRARY_PATH' >> ~/.bash_profile

#Install PV requirements
brew install openmpi
brew install gdal

#Check out PV and PVSystemTests
cd ~
mkdir workspace
cd workspace
svn co http://svn.code.sf.net/p/petavision/code/trunk PetaVision
svn co http://svn.code.sf.net/p/petavision/code/PVSystemTests PVSystemTests
cp PetaVision/docs/cmake/CMakeLists.txt .

#Install cuda and drivers
#This is a GUI prompt, TODO make this automatic
cd ~
wget http://developer.download.nvidia.com/compute/cuda/6_5/rel/installers/cuda_6.5.14_mac_64.pkg
open -W cuda_6.5.14_mac_64.pkg

#TODO get cudnn automatically
#Get cudnn
#cd ~
#wget https://developer.nvidia.com/rdp/assets/cudnn-65-osx-r2
#tar -xvf cudnn-6.5-osx-R2-rc1.tgz

#CMake and compile PetaVision
cd ~/workspace/
#cmake -DCMAKE_BUILD_TYPE=Release -DCUDA_GPU=True -DCUDA_RELEASE=True -DCUDNN=True -DCUDNN_PATH=~/cudnn -DOPEN_MP_THREADS=True -DPV_DIR=~/workspace/PetaVision
#cd ~/workspace/PetaVision
#make -j8
