#! /bin/bash
OPENMPIC=openmpicc
OPENMPICXX=openmpic++
MPI_HOME=/opt/local
for proj in $(ls */CMakeLists.txt | xargs -n 1 dirname)
do
    (cd $proj && cmake -DCMAKE_C_COMPILER=$OPENMPIC \
                       -DCMAKE_CXX_COMPILER=$OPENMPICXX \
                       -DMPI_C_COMPILER=$OPENMPIC \
                       -DMPI_CXX_COMPILER=$OPENMPICXX \
                       -DMPI_HOME=/opt/local \
                       -DCMAKE_BUILD_TYPE=Debug -DPV_WRKSPC_DIR=$PWD/../.. &&
                 make clean && make)
done
