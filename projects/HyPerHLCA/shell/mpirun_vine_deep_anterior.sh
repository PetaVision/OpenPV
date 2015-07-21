#!/bin/bash
MPIHOME=/usr/lib64/openmpi
MPIRUN=${MPIHOME}/bin/mpirun
BASEDIR=/home/gkenyon/workspace/HyPerHLCA2
COMMAND="${BASEDIR}/Debug/HyPerHLCA2 -p ${BASEDIR}/input/HyPerHLCA_vine_deep.params -rows 12 -columns 10"
LOGDIR=/nh/compneuro/Data/vine/LCA/2013_02_01/output_2013_02_01_12x12x128_lambda_05X2_deep
LOGFILE=${LOGDIR}/LCA_vine_2013_02_01_12x12x128_lambda_05X2_deep.log
mkdir -p ${LOGDIR}
touch ${LOGFILE}
echo ${LOGFILE}
time ${MPIRUN} --byslot --hostfile ~/.mpi_anterior_hosts_custom -np 120 ${COMMAND} &> ${LOGFILE}
