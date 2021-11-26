#! /usr/bin/env bash

# 16 MPI Processes with M-to-N communication        #
# MPI quilt has 1 row, 4 columns, 4 batch processes #
# MPI block has 1 row, 2 columns, 2 batch processes #

test -e output && rm -r output

export MPICOMMAND="mpiexec -n 16 --oversubscribe"
export RUNNAME=MtoN
export RUNDESC="16 processes, M-to-N"
export OUTPUTPVPTRUNC=3344048

scriptdir="$(dirname ${0})"
bash "${scriptdir}/testProbeOutput.bash"