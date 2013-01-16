#!/bin/bash
MPIHOME=/usr/lib64/openmpi
MPIRUN=${MPIHOME}/bin/mpirun
BASEDIR=/home/gkenyon/workspace/MPITest2
COMMAND="${BASEDIR}/Debug/MPITest2 -p ${BASEDIR}/input/MPI_test.params -rows 8 -columns 8"
LOGFILE=/nh/compneuro/Data/MPITest2/MPI_test.log
time ${MPIRUN} --byslot --hostfile ~/.mpi_hosts -np 64 ${COMMAND} 1> ${LOGFILE}
