#! /usr/bin/env bash

# A script to run the ProbeOutputTest executable under various params files,
# and compare the probe output files to correct files.
# The script uses compareProbeOutput.py, which includes a tolerance for
# floating-point round-off effects.

# This script requires several environmental variables to be set.
# MPICOMMAND     is the mpiexec or mpirun command, together with options, e.g. "mpiexec -n 4"
#                It can be empty for single process runs.
# PVEXECUTABLE   The PetaVision program, typically Debug/ProbeOutputTest or Release/ProbeOutputTest
# RUNNAME        is the name of the run, appearing in the config filename and output directories.
#                Currently, the allowable RUNNAME values are "oneproc", "batchMPI", "MtoN", "MtoN2".
# RUNDESC        is a brief description of the run, used in informational messages.
# OUTPUTPVPTRUNC is the number of bytes to truncate Output.pvp to, when restarting or initializing
#                from checkpoint, in order to simulate restarting from a previous interruped run.

function toldiff() {
   python3 input/compareProbeOutput.py "$@";
}

function toldiffloop() {
    status=0
    correctdir="input/correctProbeOutput/${1}"
    for k in {0..7}
    do
        targetfile="OutputL2Norm_batchElement_${k}.txt"
        targetpath="$(find output -name ${targetfile})"
        correctfile="${correctdir}/correct_OutputL2Norm_${k}.txt"
        if test "$(echo "$targetpath" | wc -l)" -ne 1
        then
            >&2 echo "${RUNDESC}, file \"${targetfile}\" not found"
            status=1
        else
            toldiff "$correctfile" "$targetpath" 3
            if test $? -ne 0 ; then status=1 ; fi
        fi
    done
    for k in {0..7}
    do
        targetfile="TotalEnergy_batchElement_${k}.txt"
        targetpath="$(find output -name ${targetfile})"
        correctfile="${correctdir}/correct_TotalEnergy_${k}.txt"
        if test "$(echo "$targetpath" | wc -l)" -ne 1
        then
            >&2 echo "${RUNDESC}, file \"${targetfile}\" not found"
            status=1
        else
            toldiff "$correctfile" "$targetpath" 2
            if test $? -ne 0 ; then status=1 ; fi
        fi
    done
    return $status
}

type python3
if test $? -ne 0
then
    exit 1
fi

if test -z "${PVEXECUTABLE}"
then
    >&2 echo "${0} requires the PVEXECUTABLE environment variable"
    exit 1
fi

echo "RUNNAME is ${RUNNAME}"

test -e output && rm -r output

### Base run ###
echo "${RUNDESC} test, base run..."
test -e output_${RUNNAME} && rm -r output_${RUNNAME}
command="${MPICOMMAND} ${PVEXECUTABLE} input/config_${RUNNAME}.txt"
echo "Executing command:"
echo "$command"
$command
mv -i ProbeOutputTest_${RUNNAME}*.log output/
toldiffloop base > output/ProbeOutputTest_differences.log 2>&1
status="$?"

mv -i output output_${RUNNAME}

if test $status -ne 0
then
    >&2 echo "${RUNDESC}, base run failed."
    exit 1 # If this run doesn't work, no other runs would be expected to work either
fi
test $status -eq 0 && echo "${RUNDESC}, base run passed."

failure=0

### Initializing from checkpoint ###
echo "${RUNDESC} test, initializing from checkpoint..."
if test -d initializing_checkpoint ; then rm -r initializing_checkpoint ; fi
cp -pr output_${RUNNAME}/checkpoints/Checkpoint040 initializing_checkpoint
# New, experimental code
cp -pr output_${RUNNAME} output
find output -type f -delete
find output_${RUNNAME} -name TotalEnergy_batchElement_\*.txt \! -path \*filepos\* |
while read f
do
    copyname="$(echo "$f" | sed -e '1,$s|^output_'"${RUNNAME}"'/|output/|')"
    cp -p "$f" "$copyname"
    echo "There's no fire in the fireplace" >> "$copyname"
    echo "There's no carpet on the floor" >> "$copyname"
    echo "Don't try to order dinner there's no kitchen any more" >> "$copyname"
done
#cp -pr output_${RUNNAME} output
#rm output/*.log
#for k in {0..7}
#do
#    targetfile=OutputL2Norm_batchElement_${k}.txt
#    filepospath="$(find output/checkpoints/Checkpoint040 -name "${targetfile}_filepos_FileStreamRead.txt")"
#    truncat="$(cat ${filepospath})"
#    truncat=$(($truncat + 100))
#    targetpath="$(find output -name "${targetfile}")"
#    truncate -s $truncat "${targetpath}"
#    echo "xxxxxxxxxx" >> "${targetpath}"
#done
#find output -name Output.pvp |
#while read f
#do
#    truncate -s ${OUTPUTPVPTRUNC} "$f"
#done
command="${MPICOMMAND} ${PVEXECUTABLE} input/config_${RUNNAME}-ifcp.txt"
echo "Executing command:"
echo "$command"
$command
mv -i ProbeOutputTest_${RUNNAME}_initfromchkpt*.log output/
toldiffloop initfromchkpt > output/ProbeOutputTest_differences.log 2>&1
if test $? -ne 0
then
    >&2 echo "${RUNDESC}, initializing from checkpoint failed."
    failure=1
    #No need to exit here, since subsequent runs don't depend on initializing from checkpoint
else
    echo "${RUNDESC}, initializing from checkpoint passed."
fi
test -e output_${RUNNAME}_initfromchkpt && rm -r output_${RUNNAME}_initfromchkpt
mv -i output output_${RUNNAME}_initfromchkpt

### Restarting from checkpoint ###
echo "${RUNDESC} test, restarting from checkpoint..."
cp -pr output_${RUNNAME} output
rm output/*.log
for k in {0..7}
do
    targetfile=OutputL2Norm_batchElement_${k}.txt
    filepospath="$(find output/checkpoints/Checkpoint040 -name "${targetfile}_filepos_FileStreamRead.txt")"
    truncat="$(cat "${filepospath}")"
    truncat=$(($truncat + 100))
    targetpath="$(find output -name "${targetfile}")"
    truncate -s $truncat "${targetpath}"
    echo "xxxxxxxxxx" >> "${targetpath}"
done
find output -name Output.pvp |
while read f
do
    truncate -s ${OUTPUTPVPTRUNC} "$f"
done
command="${MPICOMMAND} ${PVEXECUTABLE} input/config_${RUNNAME}-restartfromchkpt.txt"
echo "Executing command:"
echo "$command"
$command
mv -i ProbeOutputTest_${RUNNAME}_restartfromchkpt*.log output/
toldiffloop restartfromchkpt > output/ProbeOutputTest_differences.log 2>&1
if test $? -ne 0
then
    >&2 echo "${RUNDESC}, restarting from checkpoint failed."
    failure=1
    #No need to exit here, since subsequent runs don't depend on restarting from checkpoint
else
    echo "${RUNDESC}, restarting from checkpoint passed."
fi
test -e output_${RUNNAME}_restartfromchkpt && rm -r output_${RUNNAME}_restartfromchkpt
mv -i output output_${RUNNAME}_restartfromchkpt

### Restarting from end ###
echo "${RUNDESC} test, restarting from end..."
cp -pr output_${RUNNAME} output
rm output/*.log
command="${MPICOMMAND} ${PVEXECUTABLE} input/config_${RUNNAME}-restartfromend.txt"
echo "Executing command:"
echo "$command"
$command

mv -i ProbeOutputTest_${RUNNAME}_restartfromend*.log output/
toldiffloop restartfromend > output/ProbeOutputTest_differences.log 2>&1
if test $? -ne 0
then
    >&2 echo "${RUNDESC}, restarting from end failed."
    failure=1
    #No need to exit here, since subsequent runs don't depend on restarting from end
else
    echo "${RUNDESC}, restarting from end passed."
fi
test -e output_${RUNNAME}_restartfromend && rm -r output_${RUNNAME}_restartfromend
mv -i output output_${RUNNAME}_restartfromend

if test "$failure" -eq 0
then
    echo "Test passed."
else
    >&2 echo "Test failed."
fi

exit "$failure"
