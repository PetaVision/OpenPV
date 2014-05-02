#! /usr/bin/env bash

# Runs CMake for PetaVision and all directories in the same directory as PetaVision/
# (that is, it assumes projects are organized as in an Eclipse Workspace)
# It assumes you want to run with MPI on.
#
# Set environment variables C_COMPILER and CXX_COMPILER to choose the compilers.
# The default compilers are the results of the commands `which mpicc` and `which mpicxx`
# If these compilers are not found and the compilers are not specified manually,
# exits with an error.
#
# Set environment variable BUILD_TYPE to control the CMAKE_BUILD_TYPE variable.
# Default is "Debug"

# Set Open MPI commands
status=0
if test -z "$C_COMPILER"
then
    C_COMPILER="$(which mpicc)"
    if test -z "$C_COMPILER"
    then
        echo "$0 Error: mpicc not found.  Set environmental variable C_COMPILER to the full path to mpicc." > /dev/stderr
        status=1
    fi
fi
if test -z "$CPP_COMPILER"
then
    CPP_COMPILER="$(which mpicxx)"
    if test -z "$C_COMPILER"
    then
        echo "$0 Error: mpicxx not found.  Set environmental variable C_COMPILER to the full path to mpicxx." > /dev/stderr
        status=1
    fi
fi
if test "$status" -ne 0
then
    exit "$status"
fi

if test -z "$BUILD_TYPE"
then
    BUILD_TYPE="Debug"
fi

function cmakecmd ()
{
    echo cmake -DMPI_C_COMPILER="$C_COMPILER" -DMPI_CXX_COMPILER="$CPP_COMPILER" $*
    cmake -DMPI_C_COMPILER="$C_COMPILER" -DMPI_CXX_COMPILER="$CPP_COMPILER" $*
}

# If called from a directory other than PetaVision/scripts, change to PetaVision/scripts
if test "${0%/*}" != "$0"
then
    cd "${0%/*}"
fi
cd ../.. # We should now be in the eclipse workspace directory
wd=$PWD
if test -z "$PV_DIR"
then
    PV_DIR="$wd/PetaVision"
fi

fails=""
nomakefile=""

# Building PetaVision does not automatically build the parser files created by flex/bison
echo ; echo ======== Building io/parser ========
cd PetaVision/src/io/parser
make all
if test "$?" -ne 0
then
    fails="$fails io/parser"
fi
cd $wd

# PetaVision must be compiled before any projects that depend on it
echo ; echo ======== Building PetaVision ========
cd PetaVision
rm -rf CMakeFiles CMakeCache.txt
cmakecmd -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
make -j4 all
if test "$?" -ne 0
then
    echo "PetaVision failed to build."
    exit 1;
fi
cd $wd

# The PetaVision/tools directory has the source code for the command-line tool readpvpheader
echo ; echo ======== Building PetaVision tools ========
cd PetaVision/tools
make
if test "$?" -ne 0
then
    fails="$fails PetaVision/tools"
fi
cd $wd

projectlist=$(ls -F | egrep '/$' | sed -e '1,$s/\///' | egrep -v '^(CMakeFiles|PetaVision)')

# Compile each project in workspace directory except PetaVision
for k in $projectlist
do
    echo ; echo ======== Building $(basename $k) ========
    if test -f $k/CMakeLists.txt
    then
        hascmake=1
        cd $k
        rm -rf CMakeFiles CMakeCache.txt
        cmakecmd -DCMAKE_BUILD_TYPE="$BUILD_TYPE" -DPV_DIR="$PV_DIR"
        make clean
        make -j4 all
        if test "$?" -ne 0
        then
            fails="$fails $k"
        fi
    else
        nocmakefile="$nocmakefile $k"
    fi
    cd $wd
done

if test -n "$fails"
then
    echo "The following projects failed to build:$fails"
else
    echo "All builds succeeded."
fi
if test -n "$nomakefile"
then
    echo "The following projects have no CMakeLists.txt:$nocmakefile"
fi
