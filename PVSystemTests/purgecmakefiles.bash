#! /bin/bash
for proj in $(ls */CMakeLists.txt | xargs -n 1 dirname)
do
    (cd $proj && rm -rf CMakeCache.txt CMakeFiles)
done
