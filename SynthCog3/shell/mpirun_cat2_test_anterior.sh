#!/bin/bash
MPIHOME=/usr/lib64/openmpi
MPIRUN=${MPIHOME}/bin/mpirun
BASEDIR=/home/gkenyon/workspace/SynthCog3
COMMAND="${BASEDIR}/Debug/SynthCog3 -p ${BASEDIR}/input/CatVsNoCatDog/Test/cat2/canny3way/CatVsNoCatDog_Test_cat2_canny3way.params -rows 4 -columns 6"
LOGFILE=/nh/compneuro/Data/ImageNet/PetaVision/CatVsNoCatDog/Test/activity/cat2/canny3way/CatVsNoCatDog_Test_cat2_canny3way.log
time ${MPIRUN} --bynode --hostfile ~/.mpi_anterior_hosts_first_half -np 24 ${COMMAND} 1> ${LOGFILE}
