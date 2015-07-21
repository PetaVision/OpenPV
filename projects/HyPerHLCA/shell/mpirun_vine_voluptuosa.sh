#!/bin/bash
MPIHOME=/usr/lib64/openmpi
MPIRUN=${MPIHOME}/bin/mpirun
MPIARGS="--mca btl tcp,self"
BASEDIR=/home/gkenyon/workspace/HyPerHLCA2
COMMAND="${BASEDIR}/Debug/HyPerHLCA2 -p ${BASEDIR}/input/HyPerHLCA_vine_noPulvinar.params -rows 6 -columns 10"
LOGDIR=/nh/compneuro/Data/vine/LCA/2013_01_29/output_2013_01_29_12x12x128_lambda_05X2_noPulvinar
LOGFILE=${LOGDIR}/LCA_vine_2013_01_29_12x12x128_lambda_05X2_noPulvinar.log
mkdir -p ${LOGDIR}
touch ${LOGFILE}
echo ${LOGFILE}
#time ${MPIRUN} --byslot -np 60 ${MPIARGS} ${COMMAND} &> ${LOGFILE}
time ${MPIRUN} -np 60 --mca btl tcp,self ${COMMAND} &> ${LOGFILE}