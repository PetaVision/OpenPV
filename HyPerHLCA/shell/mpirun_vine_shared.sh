#!/bin/bash
MPIHOME=/usr/lib64/openmpi
MPIRUN=${MPIHOME}/bin/mpirun
BASEDIR=/home/gkenyon/workspace/HyPerHLCA2
COMMAND="${BASEDIR}/Debug/HyPerHLCA2 -p ${BASEDIR}/input/HyPerHLCA_vine.params -rows 6 -columns 5"
LOGDIR=/nh/compneuro/Data/vine/LCA/2013_01_31/output_16x16x4096_NoOverlap_lambda_05X2
LOGFILE=${LOGDIR}/LCA_vine_2013_01_31.log
mkdir -p ${LOGDIR}
touch ${LOGFILE}
echo ${LOGFILE}
time ${MPIRUN} --byslot -np 30 ${COMMAND} &> ${LOGFILE}
