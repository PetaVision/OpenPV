#! /usr/bin/env bash

function toldiff() {
   python input/compareProbeOutput.py "$@";
}
test -e output && rm -r output

######## Single process runs ########

### One process, base run ###
echo "Testing one process, base run..."
test -e output_oneproc && rm -r output_oneproc
Debug/ProbeOutputTest input/config_oneproc.txt
mv -i BasicSystemTest*.log output/
status=0
for k in {0..7}
do
    toldiff input/correctProbeOutput/base/correct_${k}.txt output/OutputL2Norm_batchElement_${k}.txt || status=1
done

mv -i output output_oneproc

if test $status -ne 0
then
    >&2 echo "One process, base run failed."
    exit 1 # If this run doesn't work, no others would be expected to work either
fi
test $status -eq 0 && echo "One process, base run passed."

failure=0

### One process, initializing from checkpoint ###
echo "Testing one process, initializing from checkpoint..."
cp -pr output_oneproc output
rm output/*.log
for k in {0..7}
do
    targetfile=OutputL2Norm_batchElement_${k}.txt
    truncat="$(cat output/checkpoints/Checkpoint040/${targetfile}_filepos_FileStreamRead.txt)"
    truncat=$(($truncat + 100))
    truncate -s $truncat output/$targetfile
    echo "xxxxxxxxxx" >> output/$targetfile
done
truncate -s 11012816 output/Output.pvp
Debug/ProbeOutputTest input/config_oneproc-ifcp.txt
mv -i BasicSystemTest-ifcp*.log output/
status=0
for k in {0..7}
do
    toldiff input/correctProbeOutput/initfromchkpt/correct_${k}.txt output/OutputL2Norm_batchElement_${k}.txt || status=1
done
if test $status -ne 0
then
    echo "$diffs"
    >&2 echo "One process, initializing from checkpoint failed."
    failure=1
    #No need to exit here, since subsequent runs don't depend on initializing from checkpoint
else
    echo "One process, initializing from checkpoint passed."
fi
test -e output_oneproc_initfromchkpt && rm -r output_oneproc_initfromchkpt
mv -i output output_oneproc_initfromchkpt

### One process, restarting from checkpoint ###
echo "Testing one process, restarting from checkpoint..."
cp -pr output_oneproc output
rm output/*.log
for k in {0..7}
do
    targetfile=OutputL2Norm_batchElement_${k}.txt
    truncat="$(cat output/checkpoints/Checkpoint040/${targetfile}_filepos_FileStreamRead.txt)"
    truncat=$(($truncat + 100))
    truncate -s $truncat output/$targetfile
    echo "xxxxxxxxxx" >> output/$targetfile
done
truncate -s 11012816 output/Output.pvp
Debug/ProbeOutputTest input/config_oneproc-restartfromchkpt.txt
mv -i BasicSystemTest-restart*.log output/
status=0
for k in {0..7}
do
    toldiff input/correctProbeOutput/restartfromchkpt/correct_${k}.txt output/OutputL2Norm_batchElement_${k}.txt || status=1
done
if test $status -ne 0
then
    echo "$diffs"
    >&2 echo "One process, restarting from checkpoint failed."
    failure=1
    #No need to exit here, since subsequent runs don't depend on restarting from checkpoint
else
    echo "One process, restarting from checkpoint passed."
fi
test -e output_oneproc_restartfromchkpt && rm -r output_oneproc_restartfromchkpt
mv -i output output_oneproc_restartfromchkpt

### One process, restarting from end ###
echo "Testing one process, restarting from end..."
cp -pr output_oneproc output
rm output/*.log
Debug/ProbeOutputTest input/config_oneproc-restartfromend.txt
mv -i BasicSystemTest-restart*.log output/
stauts=0
for k in {0..7}
do
    toldiff input/correctProbeOutput/restartfromend/correct_${k}.txt output/OutputL2Norm_batchElement_${k}.txt || status=1
done
if test $status -ne 0
then
    echo "$diffs"
    >&2 echo "One process, restarting from end failed."
    failure=1
    #No need to exit here, since subsequent runs don't depend on restarting from end
else
    echo "One process, restarting from end passed."
fi
test -e output_oneproc_restartfromend && rm -r output_oneproc_restartfromend
mv -i output output_oneproc_restartfromend

if test "$failure" -eq 0
then
    echo "Test passed."
else
    >&2 echo "Test failed."
fi

exit "$failure"
