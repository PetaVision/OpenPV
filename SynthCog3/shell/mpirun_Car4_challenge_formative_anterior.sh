#!/bin/bash
MPIHOME=/usr/lib64/openmpi
MPIRUN=${MPIHOME}/bin/mpirun
BASEDIR=/home/gkenyon/workspace/SynthCog3
COMMAND="${BASEDIR}/Debug/SynthCog3 -p ${BASEDIR}/input/Heli/Challenge/Car4/canny3way2X2F/026/Heli_Challenge_Car4_canny3way2X2F.params -rows 6 -columns 16"
LOGFILE=/nh/compneuro/Data/repo/neovision-programs-petavision/Heli/Challenge/activity/Car4/canny3way2X2F/026/Heli_Challenge_Car4_canny3way2X2F.log
time ${MPIRUN} --bynode --hostfile ~/.mpi_hosts -np 96 ${COMMAND} &> ${LOGFILE}
