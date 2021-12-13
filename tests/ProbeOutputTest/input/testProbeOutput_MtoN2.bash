#! /usr/bin/env bash

# 16 MPI Processes with M-to-N communication         #
# MPI quilt has 2 rows, 2 columns, 4 batch processes #
# MPI block has 1 row, 1 column, 4 batch processes   #

test -e output && rm -r output

export MPICOMMAND="mpiexec -n 16 --oversubscribe"
export PVEXECUTABLE="${1}"
export RUNNAME=MtoN2
export RUNDESC="16 processes, M-to-N test 2"
export OUTPUTPVPTRUNC=3345680

scriptdir="$(dirname ${0})"
bash "${scriptdir}/testProbeOutput.bash"
