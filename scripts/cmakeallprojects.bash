#! /usr/bin/env bash

# If called from a directory other than PetaVision/scripts, change to PetaVision/scripts
if test "${0%/*}" != "$0"
then
    cd "${0%/*}"
fi
cd ../.. # We should now be in the eclipse workspace directory
wd=$PWD

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
rm -rf CMakeFiles
cmake -DCMAKE_C_COMPILER=openmpicc -DCMAKE_CXX_COMPILER=openmpic++ -DMPI_C_COMPILER=openmpicc -DMPI_CXX_COMPILER=openmpic++ -DCMAKE_BUILD_TYPE=Debug
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
        rm -rf CMakeFiles
        # TODO: A command line argument should be able to override the setting for PV_DIR
        cmake -DCMAKE_C_COMPILER=openmpicc -DCMAKE_CXX_COMPILER=openmpic++ -DMPI_C_COMPILER=openmpicc -DMPI_CXX_COMPILER=openmpic++ -DCMAKE_BUILD_TYPE=Debug -DPV_DIR=$PWD/../PetaVision
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

# Compile the unit tests
echo ; echo ======== Building PetaVision/tests ========
cd PetaVision/tests
make clean
make -j4 all
if test "$?" -ne 0
then
    fails="$fails tests"
fi
cd $wd
echo cd $wd

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
