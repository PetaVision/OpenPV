#!/usr/bin/env bash
mpiexec -np 4 --bind-to none Release/PASCAL_Classification -p paramsfiles/square.params -t 8 -rows 2 -columns 2 2>&1 | tee landscape.log
