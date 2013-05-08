#!/bin/bash
MPIHOME=/usr/lib64/openmpi
MPIRUN=${MPIHOME}/bin/mpirun
BASEDIR=/home/gkenyon/workspace/SynthCog3
COMMAND="${BASEDIR}/Debug/SynthCog3 -p ${BASEDIR}/input/Heli/Formative/NotCar4/canny3way2X2/Heli_Formative_NotCar4_canny3way2X2.params -rows 12 -columns 8"
LOGFILE=/nh/compneuro/Data/repo/neovision-programs-petavision/Heli/Formative/activity/NotCar4/canny3way2X2/Heli_Formative_NotCar4_canny3way2X2.log
time ${MPIRUN} --bynode --hostfile ~/.mpi_anterior_hosts -np 96 ${COMMAND} &> ${LOGFILE}
