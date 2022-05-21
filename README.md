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
# Debug build (Release is the default)
cmake -DCMAKE_BUILD_TYPE:String=Debug ../OpenPV

# If CUDA is installed but you don't want CUDA support
cmake -DPV_USE_CUDA:Bool=OFF ../OpenPV

# To specify a particular CUDA architecture
cmake -DPV_CUDA_ARCHITECTURE:String=[arch]
# [arch] can be a compute capability e.g. 7.0, an architecture name e.g. Volta,
# "Auto" to detect and use the compute capability of the current GPU,
# or left blank to use the default choice for the CUDA version

# Build with clang address sanitization
cmake -DPV_ADDRESS_SANITIZE:Bool=ON ../OpenPV
~~~~~~~~~~~~~~~~~~~~

####Running the system tests:
~~~~~~~~~~~~~~~~~~~~{.sh}
cd tests
ctest
~~~~~~~~~~~~~~~~~~~~

#####Our webpage is <http://petavision.github.io/>.
#####More detailed documentation is available at <http://petavision.github.io/doxygen>.
#####For general questions and discussion, post to our Gitter page: <https://gitter.im/PetaVision/OpenPV>
