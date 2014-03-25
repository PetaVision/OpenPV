#! /bin/bash
for proj in $(ls */CMakeLists.txt | xargs -n 1 dirname)
do
    (cd $proj && make clean && make)
done
if test -d UnitTests
then
    PV_DIR=$PWD/../PetaVision
    if test -n "$(echo "$1" | egrep '^-DPV_DIR=')"
    then
        PV_DIR="$(echo "$1" | sed -e '1,$s/^-DPV_DIR=//')"
        if test -z "$(echo "$PV_DIR" | egrep '^/')"
        then
            PV_DIR="$PWD/$PV_DIR"
        fi
    fi
    echo "Compiling UnitTests with PetaVision directory set to"
    echo "$PV_DIR"
    (cd UnitTests && make clean && make PV_DIR=$PV_DIR)
fi
