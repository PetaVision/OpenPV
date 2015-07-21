#!/bin/bash
MPIHOME=/usr/lib64/openmpi
MPIRUN=${MPIHOME}/bin/mpirun
BASEDIR=/home/gkenyon/workspace/HyPerHLCA2
COMMAND="${BASEDIR}/Debug/HyPerHLCA2 -p ${BASEDIR}/input/HyPerHLCA_vine_16x16x1024_overlapp_detail.params -rows 4 -columns 4"
LOGDIR=/nh/compneuro/Data/vine/LCA/detail/output_16x16x1024_overlap_lambda_05X2_errorthresh_005
LOGFILE=${LOGDIR}/LCA_vine_detail_16x16x1024_lambda_05X2_errorthresh_005.log
mkdir -p ${LOGDIR}
touch ${LOGFILE}
echo ${LOGFILE}
time ${MPIRUN} --byslot -np 16 ${COMMAND} &> ${LOGFILE}
