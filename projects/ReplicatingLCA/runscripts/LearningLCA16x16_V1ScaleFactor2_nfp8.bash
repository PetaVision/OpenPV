#!/bin/bash
HOSTFILE=nodes34-45.txt
JOBNAME=LearningLCA16x16_V1ScaleFactor2_nfp8
MPICOLS=16
MPIROWS=9
MPIPROCS=$(($MPICOLS*$MPIROWS))
MPIHOME=/usr/lib64/openmpi
MPIRUN=${MPIHOME}/bin/mpiexec
BASEDIR=/home/pschultz/ReplicatingLCA
COMMAND="${BASEDIR}/Release/ReplicatingLCA -p ${BASEDIR}/input/${JOBNAME}.params -rows $MPIROWS -columns $MPICOLS"
LOGDIR="${BASEDIR}/log"
LOGFILE="${LOGDIR}/${JOBNAME}.log"
mkdir -p ${LOGDIR}

echo ${LOGFILE}
starttime=$(date)
time ${MPIRUN} --bynode --hostfile "${BASEDIR}/runscripts/${HOSTFILE}" --mca btl self,tcp -np $MPIPROCS ${COMMAND} 2>&1 | tee ${LOGFILE}
stoptime=$(date)
echo "Started at $starttime" >> ${LOGFILE}
echo "Stopped at $stoptime" >> ${LOGFILE}
