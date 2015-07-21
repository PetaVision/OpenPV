#!/bin/bash
MPIHOME=/usr/lib64/openmpi
MPIRUN=${MPIHOME}/bin/mpirun
MPIARGS=""##"--mca bta tcp,self"
BASEDIR=/home/gkenyon/workspace/HyPerHLCA2
COMMAND="${BASEDIR}/Debug/HyPerHLCA2 -p ${BASEDIR}/input/HyPerHLCA_KITTI.params -rows 5 -columns 12"
LOGDIR=/nh/compneuro/Data/KITTI/LCA/2011_09_26_drive_0005_sync/
LOGFILE=${LOGDIR}/KITTI_LCA_2011_09_26_drive_0005_sync.log
mkdir -p ${LOGDIR}
touch ${LOGFILE}
echo ${LOGFILE}
time ${MPIRUN} --byslot -np 60 ${COMMAND} &> ${LOGFILE}
#time ${MPIRUN} -np 60 --mca btl tcp,self ${COMMAND} &> ${LOGFILE}