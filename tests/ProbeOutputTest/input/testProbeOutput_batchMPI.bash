#! /usr/bin/env bash

function toldiff() {
   python input/compareProbeOutput.py "$@";
}
test -e output && rm -r output

######## 4 MPI Processes in the batch dimension ########

### Four processes, batch MPI, base run ###
echo "Testing four processes, batch MPI, base run..."
test -e output_batchMPI && rm -r output_batchMPI
mpiexec -n 4 Debug/ProbeOutputTest input/config_batchMPI.txt
mv -i BasicSystemTest*.log output/
status=0
for k in {0..7}
do
    toldiff input/correctProbeOutput/base/correct_${k}.txt output/block*/OutputL2Norm_batchElement_${k}.txt || status=1
done

mv -i output output_batchMPI

if test -n "$diffs"
then
    >&2 echo "Four processes, batch MPI, base run failed."
    exit 1 # If this run doesn't work, no other runs of 4 processes and higher would be expected to work either
fi
test -z "$diffs" && echo "Four processes, batch MPI, base run passed."

failure=0

### Four processes, batch MPI, initializing from checkpoint ###
echo "Testing four processes, batch MPI, initializing from checkpoint..."
cp -pr output_batchMPI output
rm output/*.log
for k in {0..7}
do
    targetfile=OutputL2Norm_batchElement_${k}.txt
    truncat="$(cat output/checkpoints/Checkpoint040/block_*/${targetfile}_filepos_FileStreamRead.txt)"
    truncat=$(($truncat + 100))
    truncate -s $truncat output/block_*/$targetfile
    echo "xxxxxxxxxx" >> output/block_*/$targetfile
done
truncate -s 5506448 output/Output.pvp
mpiexec -n 4 Debug/ProbeOutputTest input/config_batchMPI-ifcp.txt
mv -i BasicSystemTest-ifcp*.log output/
status=0
for k in {0..7}
do
    toldiff input/correctProbeOutput/initfromchkpt/correct_${k}.txt output/block_*/OutputL2Norm_batchElement_${k}.txt || status=1
done
if status -ne 0
then
    >&2 echo "Four processes, batch MPI, initializing from checkpoint failed."
    failure=1
    #No need to exit here, since subsequent runs don't depend on initializing from checkpoint
else
    echo "Four processes, batch MPI, initializing from checkpoint passed."
fi
test -e output_batchMPI_initfromchkpt && rm -r output_batchMPI_initfromchkpt
mv -i output output_batchMPI_initfromchkpt

### Four processes, batch MPI, restarting from checkpoint ###
echo "Testing four processes, batch MPI, restarting from checkpoint..."
cp -pr output_batchMPI output
rm output/*.log
for k in {0..7}
do
    targetfile=OutputL2Norm_batchElement_${k}.txt
    truncat="$(cat output/checkpoints/Checkpoint040/block_*/${targetfile}_filepos_FileStreamRead.txt)"
    truncat=$(($truncat + 100))
    truncate -s $truncat output/block_*/$targetfile
    echo "xxxxxxxxxx" >> output/block_*/$targetfile
done
truncate -s 11012816 output/Output.pvp
mpiexec -n 4 Debug/ProbeOutputTest input/config_batchMPI-restartfromchkpt.txt
mv -i BasicSystemTest-restart*.log output/
status=0
for k in {0..7}
do
    toldiff input/correctProbeOutput/restartfromchkpt/correct_${k}.txt output/block_*/OutputL2Norm_batchElement_${k}.txt || status=1
done
if status -ne 0
then
    >&2 echo "Four processes, batch MPI, restarting from checkpoint failed."
    failure=1
    #No need to exit here, since subsequent runs don't depend on restarting from checkpoint
else
    echo "Four processes, batch MPI, restarting from checkpoint passed."
fi
test -e output_batchMPI_restartfromchkpt && rm -r output_batchMPI_restartfromchkpt
mv -i output output_batchMPI_restartfromchkpt

### Four processes, batch MPI, restarting from end ###
echo "Testing four processes, batch MPI, restarting from end..."
cp -pr output_batchMPI output
rm output/*.log
mpiexec -n 4 Debug/ProbeOutputTest input/config_batchMPI-restartfromend.txt
mv -i BasicSystemTest-restart*.log output/
status=0
for k in {0..7}
do
    toldiff input/correctProbeOutput/restartfromend/correct_${k}.txt output/block_*/OutputL2Norm_batchElement_${k}.txt || status=1
done
if status -ne 0
then
    >&2 echo "Four processes, batch MPI, restarting from end failed."
    failure=1
    #No need to exit here, since subsequent runs don't depend on restarting from end
else
    echo "Four processes, batch MPI, restarting from end passed."
fi
test -e output_batchMPI_restartfromend && rm -r output_batchMPI_restartfromend
mv -i output output_batchMPI_restartfromend

if test "$failure" -eq 0
then
    echo "Test passed."
else
    >&2 echo "Test failed."
fi

exit "$failure"
