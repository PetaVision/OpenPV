#! /bin/bash
alias cmakemac="cmake CMakeLists.txt -DCMAKE_C_COMPILER=openmpicc \
                                     -DCMAKE_CXX_COMPILER=openmpic++ \
                                     -DMPI_C_COMPILER=openmpicc \
                                     -DMPI_CXX_COMPILER=openmpic++ \
                                     -DMPI_HOME=/opt/local"
for proj in $(ls */CMakeLists.txt | xargs -n 1 dirname)
do
    (cd $proj && cmakemac -DCMAKE_BUILD_TYPE=Debug -DPV_WRKSPC_DIR=../.. &&
                 make clean && make)
done
