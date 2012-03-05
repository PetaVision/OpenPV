#! /usr/bin/env bash

# If called from a directory other than PetaVision/scripts, change to PetaVision/scripts
if test "${0%/*}" != "$0"
then
    cd "${0%/*}"
fi
cd ../.. # We should now be in the eclipse workspace directory
wd=$PWD

# PetaVision must be compiled before any projects that depend on it
cd PetaVision/lib
make clean
make -j4 all
cd $wd

# Compile each project in workspace directory except PetaVision
for k in $(ls | egrep -v PetaVision)
do
    cd $k/Debug
    make clean
    make -j4 all
    cd $wd
done

# Compile the unit tests
cd PetaVision/tests
make clean
make -j4 all
cd $wd