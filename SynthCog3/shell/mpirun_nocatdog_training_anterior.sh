#!/bin/bash
MPIHOME=/usr/lib64/openmpi
MPIRUN=${MPIHOME}/bin/mpirun
BASEDIR=/home/gkenyon/workspace/SynthCog3
COMMAND="${BASEDIR}/Debug/SynthCog3 -p ${BASEDIR}/input/CatVsNoCatDog/Training/nocatdog/canny3way/CatVsNoCatDog_Training_nocatdog_canny3way.params -rows 8 -columns 8"
LOGFILE=/nh/compneuro/Data/ImageNet/PetaVision/CatVsNoCatDog/Training/activity/nocatdog/canny3way/CatVsNoCatDog_Training_nocatdog_canny3way.log
time ${MPIRUN} --bynode --hostfile ~/.mpi_anterior_hosts_second_half -np 64 ${COMMAND} 1> ${LOGFILE}
