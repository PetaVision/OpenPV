#!/bin/bash
MPIHOME=/usr/lib64/openmpi
MPIRUN=${MPIHOME}/bin/mpirun
BASEDIR=/home/gkenyon/workspace/SynthCog3
COMMAND="${BASEDIR}/Debug/SynthCog3 -p ${BASEDIR}/input/CatVsNoCatDog/Test/cat/canny3way/CatVsNoCatDog_Test_cat_canny3way.params -rows 8 -columns 8"
LOGFILE=/nh/compneuro/Data/ImageNet/PetaVision/CatVsNoCatDog/Test/activity/cat/canny3way/CatVsNoCatDog_Test_cat_canny3way.log
time ${MPIRUN} --bynode --hostfile ~/.mpi_anterior_hosts_first_half -np 64 ${COMMAND} 1> ${LOGFILE}
