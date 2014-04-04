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
if test -f Makefile
then
    hasmake=1
elif test -f lib/Makefile
then
    hasmake=1
    cd lib
else
    hasmake=0
fi
if test $hasmake -eq 1
then
    make -j4 all
    if test "$?" -ne 0
    then
        fails="$fails PetaVision"
    fi
else
    nomakefile="$nomakefile $k"
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
    if test -f $k/Makefile
    then
        hasmake=1
        cd $k
    elif test -f $k/Debug/Makefile
    then
        hasmake=1
        cd $k/Debug
    else
        hasmake=0
    fi
    if test $hasmake -eq 1
    then
        make clean
        make -j4 all
        if test "$?" -ne 0
        then
            fails="$fails $k"
        fi
    else
        nomakefile="$nomakefile $k"
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
    echo "The following projects have no makefile:$nomakefile"
fi
