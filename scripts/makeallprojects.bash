#! /usr/bin/env bash

# If called from a directory other than PetaVision/scripts, change to PetaVision/scripts
if test "${0%/*}" != "$0"
then
    cd "${0%/*}"
fi
cd ../.. # We should now be in the eclipse workspace directory
wd=$PWD

fails=""

# Building PetaVision does not automatically build the parser files created by flex/bison
cd PetaVision/src/io/parser
echo cd $PWD
make all
if test "$?" -ne 0
then
    fails="$fails io/parser"
fi
cd $wd

# PetaVision must be compiled before any projects that depend on it
cd PetaVision/lib
echo cd $PWD
make -j4 all
if test "$?" -ne 0
then
    fails="$fails PetaVision"
fi
cd $wd

# Compile each project in workspace directory except PetaVision
for k in $(ls | egrep -v PetaVision)
do
    cd $k/Debug
    echo cd $PWD
    make clean
    make -j4 all
    if test "$?" -ne 0
    then
        fails="$fails $k"
    fi
    cd $wd
done

# Compile the unit tests
cd PetaVision/tests
echo cd $PWD
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
