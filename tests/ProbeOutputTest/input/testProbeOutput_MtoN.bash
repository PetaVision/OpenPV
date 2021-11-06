#! /usr/bin/env bash

function toldiff() {
   python input/compareProbeOutput.py "$@";
}
test -e output && rm -r output

######## 16 MPI Processes with MtoN communication          ########
######## MPI quilt has 1 row, 4 columns, 4 batch processes ########
######## MPI block has 1 row, 2 columns, 2 batch processes ########

### 16 processes, M-to-N, base run ###
echo "Testing 16 processes, M-to-N, base run..."
test -e output_MtoN && rm -r output_MtoN
mpiexec -n 16 --oversubscribe Debug/ProbeOutputTest input/config_MtoN.txt
mv -i MtoN*.log output/
status=0
for k in {0..7}
do
    toldiff input/correctProbeOutput/base/correct_${k}.txt output/block*/OutputL2Norm_batchElement_${k}.txt || status=1
done

mv -i output output_MtoN

if test $status -ne 0
then
    >&2 echo "16 processes, M-to-N, base run failed."
    exit 1 # If this run doesn't work, no other 16-process runs would be expected to work either
fi
test $status -eq 0 && echo "16 processes, M-to-N, base run passed."

failure=0

### 16 processes, M-to-N, initializing from checkpoint ###
echo "Testing 16 processes, M-to-N, initializing from checkpoint..."
cp -pr output_MtoN output
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
mpiexec -n 16 --oversubscribe Debug/ProbeOutputTest input/config_MtoN-ifcp.txt
mv -i MtoN_initfromchkpt*.log output/
status=0
for k in {0..7}
do
    toldiff input/correctProbeOutput/initfromchkpt/correct_${k}.txt output/block_*/OutputL2Norm_batchElement_${k}.txt || status=1
done
if test $status -ne 0
then
    >&2 echo "16 processes, M-to-N, initializing from checkpoint failed."
    failure=1
    #No need to exit here, since subsequent runs don't depend on initializing from checkpoint
else
    echo "16 processes, M-to-N, initializing from checkpoint passed."
fi
test -e output_MtoN_initfromchkpt && rm -r output_MtoN_initfromchkpt
mv -i output output_MtoN_initfromchkpt

### 16 processes, M-to-N, restarting from checkpoint ###
echo "Testing 16 processes, M-to-N, restarting from checkpoint..."
cp -pr output_MtoN output
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
mpiexec -n 16 --oversubscribe Debug/ProbeOutputTest input/config_MtoN-restartfromchkpt.txt
mv -i MtoN_restartfromchkpt*.log output/
status=0
for k in {0..7}
do
    toldiff input/correctProbeOutput/restartfromchkpt/correct_${k}.txt output/block_*/OutputL2Norm_batchElement_${k}.txt || status=1
done
if test $status -ne 0
then
    >&2 echo "16 processes, M-to-N, restarting from checkpoint failed."
    failure=1
    #No need to exit here, since subsequent runs don't depend on restarting from checkpoint
else
    echo "16 processes, M-to-N, restarting from checkpoint passed."
fi
test -e output_MtoN_restartfromchkpt && rm -r output_MtoN_restartfromchkpt
mv -i output output_MtoN_restartfromchkpt

### 16 processes, M-to-N, restarting from end ###
echo "Testing 16 processes, M-to-N, restarting from end..."
cp -pr output_MtoN output
rm output/*.log
mpiexec -n 16 --oversubscribe Debug/ProbeOutputTest input/config_MtoN-restartfromend.txt
mv -i MtoN*.log output/
status=0
for k in {0..7}
do
    toldiff input/correctProbeOutput/restartfromend/correct_${k}.txt output/block_*/OutputL2Norm_batchElement_${k}.txt || status=1
done
if test $status -ne 0
then
    >&2 echo "16 processes, M-to-N, restarting from end failed."
    failure=1
    #No need to exit here, since subsequent runs don't depend on restarting from end
else
    echo "16 processes, M-to-N, restarting from end passed."
fi
test -e output_MtoN_restartfromend && rm -r output_MtoN_restartfromend
mv -i output output_MtoN_restartfromend

if test "$failure" -eq 0
then
    echo "Test passed."
else
    >&2 echo "Test failed."
fi

exit "$failure"
