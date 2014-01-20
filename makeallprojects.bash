#! /bin/bash
for proj in $(ls */CMakeLists.txt | xargs -n 1 dirname)
do
    (cd $proj && make clean && make)
done
