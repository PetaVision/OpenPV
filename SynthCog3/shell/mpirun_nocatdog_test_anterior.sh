#!/bin/bash
MPIHOME=/usr/lib64/openmpi
MPIRUN=${MPIHOME}/bin/mpirun
BASEDIR=/home/gkenyon/workspace/SynthCog3
COMMAND="${BASEDIR}/Debug/SynthCog3 -p ${BASEDIR}/input/CatVsNoCatDog/Test/nocatdog/canny3way/CatVsNoCatDog_Test_nocatdog_canny3way.params -rows 4 -columns 8"
LOGFILE=/nh/compneuro/Data/ImageNet/PetaVision/CatVsNoCatDog/Test/activity/nocatdog/canny3way/CatVsNoCatDog_Test_nocatdog_canny3way.log
time ${MPIRUN} --bynode --hostfile ~/.mpi_anterior_hosts_second_half -np 32 ${COMMAND} 1> ${LOGFILE}
