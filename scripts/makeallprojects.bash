#! /usr/bin/env bash

# If called from a directory other than PetaVision/scripts, change to PetaVision/scripts
if test "${0%/*}" != "$0"
then
    cd "${0%/*}"
fi
cd ../.. # We should now be in the eclipse workspace directory
wd=$PWD

# Building PetaVision does not automatically build the parser files created by flex/bison
cd PetaVision/src/io/parser
echo cd $PWD
make all
cd $wd

# PetaVision must be compiled before any projects that depend on it
cd PetaVision/lib
echo cd $PWD
make -j4 all
cd $wd

# Compile each project in workspace directory except PetaVision
for k in $(ls | egrep -v PetaVision)
do
    cd $k/Debug
    echo cd $PWD
    make clean
    make -j4 all
    cd $wd
done

# Compile the unit tests
cd PetaVision/tests
echo cd $PWD
make clean
make -j4 all
cd $wd
echo cd $wd
