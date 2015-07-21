#!/bin/bash
MPIHOME=/usr/lib64/openmpi
MPIRUN=${MPIHOME}/bin/mpirun

## FOR VOLUPTUOSA
#MPIARGS="--mca bta tcp,self"

NUMPROCS="4"
BASEDIR=/home/dpaiton/workspace/HyPerHLCA
COMMAND="${BASEDIR}/Debug/HyPerHLCA -p ${BASEDIR}/input/HyPerHLCA_CIFAR_RGB.params -rows 2 -columns 2"
LOGDIR=/nh/compneuro/Data/Cifar/LCA/data_driven/pass0
LOGFILE=${LOGDIR}/LCA_Deep_CIFAR.log
mkdir -p ${LOGDIR}
touch ${LOGFILE}
echo ${LOGFILE}
time ${MPIRUN} -np ${NUMPROCS} ${MPIARGS} ${COMMAND} &> ${LOGFILE}
