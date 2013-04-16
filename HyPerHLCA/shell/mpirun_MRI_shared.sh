#!/bin/bash
MPIHOME=/usr/lib64/openmpi
MPIRUN=${MPIHOME}/bin/mpirun
BASEDIR=/home/gkenyon/workspace_new/HyPerHLCA2
COMMAND="${BASEDIR}/Debug/HyPerHLCA2 -p ${BASEDIR}/input/HyPerHLCA_MRI.params -rows 4 -columns 4"
LOGFILE=/nh/compneuro/Data/MRI/LCA/5_subjects/MRI_LCA_5_subjects.log
time ${MPIRUN} --byslot -np 16 ${COMMAND} 1> ${LOGFILE}
