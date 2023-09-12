#! /usr/bin/env bash

# ProbeOutputTest script for single process run #

if test -z "${1}"
then
    >&2 echo "$(basename "${0}") requires the path to the PetaVision executable as an argument."
    exit 1
fi

test -e output && rm -r output

export MPICOMMAND=""
export PVEXECUTABLE="${1}"
export RUNNAME=oneproc
export RUNDESC="One process"
export OUTPUTPVPTRUNC=11012816

scriptdir="$(dirname ${0})"
bash "${scriptdir}/testProbeOutput.bash"
