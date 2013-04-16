#!/bin/bash
MPIHOME=/usr/lib64/openmpi
MPIRUN=${MPIHOME}/bin/mpirun
BASEDIR=/home/gkenyon/workspace/HyPerHLCA2
COMMAND="${BASEDIR}/Debug/HyPerHLCA2 -p ${BASEDIR}/input/HyPerHLCA_vine.params -rows 6 -columns 5"
LOGDIR=/nh/compneuro/Data/vine/LCA/2013_01_31/output
LOGFILE=${LOGDIR}/LCA_vine_2013_01_31.log
mkdir -p ${LOGDIR}
time ${MPIRUN} --byslot -np 30 ${COMMAND} 1> ${LOGFILE}
