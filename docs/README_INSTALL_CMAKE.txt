-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
PETAVISION INSTALL
-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

Before installing PetaVision, you need the following libraries:
	Open MPI
		 http://www.open-mpi.org/software/ompi/v1.6/
	GDAL
		http://trac.osgeo.org/gdal/wiki/DownloadingGdalBinaries	
	CMake
		http://www.cmake.org/cmake/resources/software.html
	SubVersion
		http://subversion.apache.org/packages.html

We recommend using apt-get on linux systems, mac ports on OSX systems, and cygwin on Windows systems to obtain the above libraries.

You will need a SourceForge account, which is free: https://sourceforge.net/user/registration

If you wish to contribute (commit changes) to PetaVision, contact us to be added as a developer.

We recommend the following, although they are not required:
	Octave
		http://www.gnu.org/software/octave/download.html
		We recommend getting the latest octave-devel version on mac ports if you are using OSX.
	Enthought Python Distribution
		https://enthought.com/products/epd/free/

-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

To install a clean build of PetaVision:

Create a workspace directory:
	mkdir workspace

Navigate into this directory:
	cd workspace

Download PetaVision from subversion:
	If you are a developer:
		svn co https://username@svn.code.sf.net/p/petavision/code/trunk PetaVision
	If you would like a read-only version:
		svn co http://svn.code.sf.net/p/petavision/code/trunk PetaVision

Download any optional sandboxes or systems tests. First, as an example, the BIDS sandbox:
	Developers: svn co https://username@svn.code.sf.net/p/petavision/code/sandbox/BIDS BIDS
	Users: svn co http://svn.code.sf.net/p/petavision/code/sandbox/BIDS BIDS

Copy the CMake project configuration file from the PetaVision docs folder to the workspace folder:
	cp PetaVision/docs/cmake/CMakeLists.txt .

Run CMake to create your make files:
    cmake CMakeLists.txt -DCMAKE_C_COMPILER=<c_compiler> -DCMAKE_CXX_COMPILER=<cpp_compiler>

    where <c_compiler> is usually your mpi compiler (e.g. mpicc or openmpicc) and <cpp_compiler> is your c++ compiler.
   
    You can also add the option -DCMAKE_BUILD_TYPE=Release or -DCMAKE_BUILD_TYPE=Debug to control debugger and optimization options.

Run the Makefile to build PetaVision and any additional sandboxes or systems tests.
	make

Done! To rebuild, run 'make clean' then 'make' in the workspace folder.

To execute a PetaVision simulation, run it from the given Debug folder. For example, if you chose to checkout the BIDS repository:
cd BIDS/
./Debug/BIDS -p input/params.pv
