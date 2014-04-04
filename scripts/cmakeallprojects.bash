#! /usr/bin/env bash

# Set Open MPI commands
if test -z "$C_COMPILER"
then
    if test "$(uname)" = "Darwin"
    then
        C_COMPILER=mpicc-openmpi-mp
    elif test "$(uname)" = "Linux"
    then
        C_COMPILER=mpicc
    fi
fi
if test -z "$CPP_COMPILER"
then
    if test "$(uname)" = "Darwin"
    then
        CPP_COMPILER=mpicxx-openmpi-mp
    elif test "$(uname)" = "Linux"
    then
        CPP_COMPILER=mpic++
    fi
fi
if test -z "$MPI_HOME"
then
    if test "$(uname)" = "Darwin"
    then
        MPI_HOME=/opt/local
    elif test "$(uname)" = "Linux"
    then
        MPI_HOME=/usr/lib64/openmpi
    fi
fi
if test -z "$BUILD_TYPE"
then
    BUILD_TYPE="Debug"
fi

function cmakecmd ()
{
    echo cmake -DCMAKE_C_COMPILER="$C_COMPILER" -DCMAKE_CXX_COMPILER="$CPP_COMPILER" -DMPI_C_COMPILER="$C_COMPILER" -DMPI_CXX_COMPILER="$CPP_COMPILER" $*
    cmake -DCMAKE_C_COMPILER="$C_COMPILER" -DCMAKE_CXX_COMPILER="$CPP_COMPILER" -DMPI_C_COMPILER="$C_COMPILER" -DMPI_CXX_COMPILER="$CPP_COMPILER" $*
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
