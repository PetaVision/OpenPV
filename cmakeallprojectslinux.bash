#! /bin/bash
for proj in $(ls */CMakeLists.txt | xargs -n 1 dirname)
do
    (cd $proj && cmakemac -DCMAKE_BUILD_TYPE=Debug -DPV_WRKSPC_DIR=../..)
done
