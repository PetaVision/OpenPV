#! /usr/bin/env bash

# ProbeOutputTest script for 4 MPI Processes in the batch dimension #

if test -z "${1}"
then
    >&2 echo "$(basename "${0}") requires the path to the PetaVision executable as an argument."
    exit 1
fi

test -e output && rm -r output

export MPICOMMAND="mpiexec -n 4"
export PVEXECUTABLE="${1}"
export RUNNAME=batchMPI
export RUNDESC="Four processes, batch MPI"
export OUTPUTPVPTRUNC=5506448

scriptdir="$(dirname ${0})"
bash "${scriptdir}/testProbeOutput.bash"
