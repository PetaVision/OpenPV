#! /usr/bin/env bash
if test "${0%/*}" != "$0"
then
    cd "${0%/*}"
fi
cd ..
wd=$PWD

cd ../PetaVision/lib
make clean
make -j4 all
cd $wd

for k in $(ls | egrep -v ../PetaVision)
do
    cd $k/Debug
    make clean
    make -j4 all
    cd $wd
done

cd ../PetaVision/tests
make clean
make -j4 all
cd $wd