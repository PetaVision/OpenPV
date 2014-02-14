#! /bin/bash
for proj in $(ls */CMakeLists.txt | xargs -n 1 dirname)
do
    (cd $proj && make clean && make)
done
if test -d UnitTests
then
    (cd UnitTests && make clean && make)
fi
