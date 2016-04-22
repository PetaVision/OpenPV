#!/usr/bin/env bash
mpiexec -np 4 --bind-to none Release/PASCAL_Classification -p paramsfiles/landscape.params -t 8 -rows 1 -columns 4 2>&1 | tee landscape.log
