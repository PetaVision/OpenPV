###PetaVision is an open source, object oriented neural simulation toolbox optimized for high-performance multi-core, multi-node computer architectures.

####Quick instructions for installing PetaVision and running the system tests:

####Required dependencies:
* gcc/g++ (https://gcc.gnu.org/).
* bison (https://www.gnu.org/software/bison/).
* flex (https://www.gnu.org/software/flex/).
* cmake (http://www.cmake.org)

####Optional (but strongly suggested) dependencies:
* OpenMPI (http://www.open-mpi.org).  
 * You'll need mpicc, mpic++, and mpiexec.
* OpenMP (http://openmp.org/wp/)
* CUDA (https://developer.nvidia.com/cuda-downloads)
* cuDNN (https://developer.nvidia.com/cudnn)

####Suggested additional tools:
* lua (http://www.lua.org/) 
 * For designing parameter files for complex networks.
* octave (https://www.gnu.org/software/octave/) 
 * To read/analyze data from OpenPV.
* python (https://www.python.org/) 
 * To read/analyze data from OpenPV.
* mermaid (http://knsv.github.io/mermaid/) 
 * For generating graphical drawings of networks from parameter files. 

####Building:
~~~~~~~~~~~~~~~~~~~~{.sh}
git clone http://github.com/PetaVision/OpenPV.git
mkdir build
cd build
cmake ../OpenPV
make
~~~~~~~~~~~~~~~~~~~~
####Other build options:
~~~~~~~~~~~~~~~~~~~~{.sh}
# If CUDA is installed but you don't want CUDA support
cmake -DPV_USE_CUDA:Bool=OFF ../OpenPV
# Build with clang address santization
cmake -DPV_ADDRESS_SANITIZE:Bool=ON ../OpenPV
# Debug build (Release is the default)
#  OpenPV must be compiled as Debug to successfully run the system tests
cmake -DCMAKE_BUILD_TYPE:String=Debug ../OpenPV
~~~~~~~~~~~~~~~~~~~~

####Running the system tests:
~~~~~~~~~~~~~~~~~~~~{.sh}
cd tests
ctest
~~~~~~~~~~~~~~~~~~~~

#####Our webpage is <http://petavision.github.io/>.
#####More detailed documentation is available at <http://petavision.github.io/doxygen>.
#####For general questions and discussion, post to our Gitter page: <https://gitter.im/PetaVision/OpenPV>
