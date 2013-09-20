#!/bin/bash
MPIHOME=/usr/lib64/openmpi
MPIRUN=${MPIHOME}/bin/mpirun
BASEDIR=/home/gkenyon/workspace/HyPerHLCA2
COMMAND="${BASEDIR}/Debug/HyPerHLCA2 -p ${BASEDIR}/input/HyPerHLCA_vine_how2catchSquirrel_deep.params -rows 4 -columns 2"
LOGDIR=/nh/compneuro/Data/vine/LCA/2013_01_24/output_2013_01_24_how2catchSquirrel_12x12x128_deep
LOGFILE=${LOGDIR}/LCA_vine_2013_01_24_how2catchSquirrel_12x12x128_deep.log
mkdir -p ${LOGDIR}
touch ${LOGFILE}
echo ${LOGFILE}
time ${MPIRUN} --byslot --hostfile ~/.mpi_anterior_hosts_anticustom -np 8 ${COMMAND} &> ${LOGFILE}
