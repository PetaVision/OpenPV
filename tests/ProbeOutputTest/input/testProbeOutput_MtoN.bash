#! /usr/bin/env bash

# 16 MPI Processes with M-to-N communication        #
# MPI quilt has 1 row, 4 columns, 4 batch processes #
# MPI block has 1 row, 2 columns, 2 batch processes #

if test -z "${1}"
then
    >&2 echo "$(basename "${0}") requires the path to the PetaVision executable as an argument."
    exit 1
fi

test -e output && rm -r output

export MPICOMMAND="mpiexec -n 16 --oversubscribe"
export PVEXECUTABLE="${1}"
export RUNNAME=MtoN
export RUNDESC="16 processes, M-to-N"
export OUTPUTPVPTRUNC=3344048

scriptdir="$(dirname ${0})"
bash "${scriptdir}/testProbeOutput.bash"
