#! /usr/bin/env bash

# ProbeOutputTest script for single process run #

test -e output && rm -r output

export MPICOMMAND=""
export PVEXECUTABLE="${1}"
export RUNNAME=oneproc
export RUNDESC="One process"
export OUTPUTPVPTRUNC=11012816

scriptdir="$(dirname ${0})"
bash "${scriptdir}/testProbeOutput.bash"
