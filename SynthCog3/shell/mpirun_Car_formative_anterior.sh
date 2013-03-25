#!/bin/bash
MPIHOME=/usr/lib64/openmpi
MPIRUN=${MPIHOME}/bin/mpirun
BASEDIR=/home/gkenyon/workspace_new/SynthCog3
COMMAND="${BASEDIR}/Debug/SynthCog3 -p ${BASEDIR}/input/Heli/Formative/Car/canny3way2X2/Heli_Formative_Car_canny3way2X2.params -rows 6 -columns 8"
LOGFILE=/nh/compneuro/Data/repo/neovision-programs-petavision/Heli/Formative/activity/Car/canny3way2X2/Heli_Formative_Car_canny3way2X2.log
time ${MPIRUN} --bynode --hostfile ~/.mpi_anterior_hosts_first_half -np 48 ${COMMAND} 1> ${LOGFILE}
