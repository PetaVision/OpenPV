#!/usr/bin/env bash
mpiexec -np 4 --bind-to none Release/PASCAL_Classification -p paramsfiles/portrait.params -t 8 -rows 4 -columns 1 2>&1 | tee landscape.log
