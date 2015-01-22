#! /usr/bin/env bash
#
# This script runs each systems test, suppressing all output but reporting
# whether the test passed or failed.  The list of system is hardcoded in the
# script; you need to add a new section if a new test is created.
#
# The script assumes that it is in the PetaVision/scripts directory.
# It can be run from any directory.
#
# When run without arguments, it assumes that the library and the tests
# were compiled with MPI.  It therefore runs system tests with both
# a single process and multiple processes.
#
# To turn off the MPI-specific tests, do "runalltests.bash --nompi"
# To set the mpirun command, do "runalltests.bash --mpirun=/path/to/mpirun"
# If mpirun is not set on the command line, it defaults to the result of
# searching the path for mpiexec, and then mpirun.
#
# To run with threads, do "runalltests.bash --threads".  This will pass the argument "-t" to the tests.
# To run with a specific number of threads, do "runalltests.bash --threads=<number>"

# Navigate to eclipse workspace directory.
if test "${0%/*}" != "$0"
then
    cd "${0%/*}"
fi
cd ../
pvdir=$PWD # $pvdir is the directory containing the PetaVision project
cd ..
workspacedir=$PWD # $workspacedir is the eclipse workspace directory

valgrindcommand=""
#valgrindcommand="valgrind --suppressions=$pvdir/valgrind/petavision-mac.supp --num-callers=50 --leak-check=full --show-leak-kinds=all --track-origins=yes"

fails=""
dne=""

function runandecho() {
    testname=$1
    shift
    logfilebasename=$1
    shift
    echo $valgrindcommand $* $threadopt
    if $valgrindcommand $* $threadopt &> ${logfilebasename}_1.log
    then
        result=passed
    else
        result=FAILED
        fails="$fails $testname/${logfilebasename}_1.log"
    fi
    echo "$testname $result (output in ${logfilebasename}_1.log)"
}

# Set default to use MPI
usempi=1
PV_MPIRUN=""

# Set default to not use threads
threadopt=""

# Check for --nompi option, or set PV_MPIRUN using --mpirun= option
for opt in "$@"
do
    if test "$opt" = "--nompi"
    then
        usempi=0
    elif test "${opt%=*}" = "--mpirun"
    then
        usempi=1
        PV_MPIRUN="${opt#*=}"
    elif test "$opt" = "--threads"
    then
        threadopt=" -t"
    elif test "${opt%=*}" = "--threads"
    then
        threadnum="${opt#*=}"
        # Sanity check $threadnum
        if test "$(echo "${threadnum}" | wc -w)" -ne 1
        then
            echo "$0 error: --threads= takes a single positive integer argument." > /dev/stderr
            echo "$0 error: Argument was \"${threadnum}\"." > /dev/stderr
            exit 1
        fi
        if test -z "$(echo "${threadnum}" | egrep '^[1-9][0-9]*$')"
        then
            echo "$0 error: --threads= takes a positive integer argument." > /dev/stderr
            echo "$0 error: Argument was \"${threadnum}\"." > /dev/stderr
            exit 1
        fi
        threadopt=" -t ${threadnum}"
    fi
done

if test "$usempi" = 0
then
    function mpirunandecho() {
        false
    }
else
    if test -z "$PV_MPIRUN"
    then
        PV_MPIRUN="$(which mpiexec)" || PV_MPIRUN="$(which mpirun)"
        if test -z "$PV_MPIRUN"
        then
            echo "$0 error: Unable to find mpiexec or mpirun." > /dev/stderr
            echo "To specify the path, use the option --mpirun=/path/to/mpirun" > /dev/stderr
            echo "To run without mpi, use the option --nompi" > /dev/stderr
            exit 1
        fi
    fi
    function mpirunandecho() {
        numprocs=$1
        shift
        testname=$1
        shift
        logfilebasename=$1
        shift
        logfilename="${logfilebasename}_${numprocs}.log"
        echo $PV_MPIRUN -np $numprocs $valgrindcommand $* $threadopt
        if $PV_MPIRUN -np $numprocs $valgrindcommand $* $threadopt &> "${logfilename}"
        then
            result=passed
        else
            result=FAILED
            fails="$fails $testname/${logfilename}"
        fi
        echo "$testname with $numprocs processes $result (output in ${logfilebasename}_$numprocs.log)"
    }
fi
function mpi_np2_np4_runandecho() {
    mpirunandecho 2 $*
    mpirunandecho 4 $*
}

testname=BasicSystemTest
arglist="-p input/BasicSystemTest.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=AdjustAxonalArborsTest
arglist="-p input/AdjustAxonalArborsTest.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=ArborSystemTest
arglist="-p input/test_arbors.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=BinningLayerTest
arglist="-p input/BinningLayerTest.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=CheckpointSystemTest
arglist=""
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=CloneHyPerConnTest
arglist="-p input/CloneHyPerConnTest.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=CloneKernelConnTest
arglist="-p input/CloneKernelConnTest.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=CloneVLayerTest
arglist=""
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=ConnectionRestartTest
arglist="-p input/ConnectionRestartTest.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=ConvertToGrayScaleTest
arglist="-p input/ConvertToGrayScaleTest.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=CopyConnTest
arglist=""
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=DatastoreDelayTest
arglist="-p input/DatastoreDelayTest.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=DelaysToFeaturesTest
arglist="-p input/test_delays.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=FourByFourGenerativeTest
arglist=""
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=FourByFourTopDownTest
arglist=""
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=GenericSystemTest
arglist="-p input/GenericSystemTest.params -c checkpoints/Checkpoint6 --testall"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

#testname=GPUSystemTest
#cd "$testname"
#arglist="-d 0 -p input/test_gpu.params"
#runandecho $testname $testname Debug/$testname $arglist
#mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
#cd "$workspacedir"
echo "TODO: fix GPUSystemTest and maybe implement GPUs"

testname=ImageSystemTest
arglist="-p input/multiframe_SystemTest.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=ImportParamsTest
arglist="-p input/ImportParamsTest.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=ImprintConnTest
arglist="-p input/ImprintConnTest.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=InitWeightsTest
arglist="-p input/test_initweights.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=KernelActivationTest
arglist=""
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=KernelTest
logfilebasename=test_kernel
arglist="-p input/test_kernel.params"
cd "$testname"
runandecho $testname $logfilebasename Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $logfilebasename Debug/$testname $arglist
cd "$workspacedir"

testname=KernelTest
logfilebasename=test_kernel_normalizepost_shrunken
arglist="-p input/test_kernel_normalizepost_shrunken.params"
cd "$testname"
runandecho $testname $logfilebasename Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $logfilebasename Debug/$testname $arglist
cd "$workspacedir"

testname=LayerPhaseTest
arglist="-p input/LayerPhaseTest.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=LayerRestartTest
arglist=""
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=LCATest
arglist="-p input/LCATest.params -c checkpoints/Checkpoint6 --testall"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=LIFTest
arglist="-p input/LIFTest.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=MarginWidthTest
arglist="-p input/MarginWidthTest.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=MatchingPursuitTest
arglist=""
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=MaxPoolTest
arglist="-p input/maxpooltest.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=MLPTest
arglist=""
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=MovieSystemTest
arglist="-p input/MovieSystemTest.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=MPITest2
arglist="-p input/MPI_test.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=NormalizeSystemTest
arglist="-p input/NormalizeSystemTest.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=ParameterSweepTest
cd "$testname"
arglist="-p input/ParameterSweepTest.params"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=PlasticConnTest
cd "$testname"
arglist="-p input/PlasticConnTest.params"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=PlasticTransposeConnTest
cd "$testname"
arglist="-p input/PlasticTransposeConnTest.params"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=RandStateSystemTest
cd "$testname"
arglist="-p input/RandStateSystemTest.params"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=ReadArborFileTest
cd "$testname"
arglist="-p input/ReadArborFileTest.params"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=ReceiveFromPostTest
cd "$testname"
arglist="-p input/postTest.params"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=ShrunkenPatchFlagTest
cd "$testname"
arglist="-p input/ShrunkenPatchFlagTest.params" # parameter filename is in main()
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=ShrunkenPatchTest
cd "$testname"
arglist="" # parameter filename is in main()
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=StochasticReleaseTest
cd "$testname"
arglist="-p input/StochasticReleaseTest.params"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=test_border_activity
cd "$testname"
arglist=""
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=test_cocirc
cd "$testname"
arglist="-p input/test_cocirc.params"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=test_constant_input
cd "$testname"
arglist="-p input/test_constant_input.params"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=test_delta
cd "$testname"
arglist=""
runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=test_delta_pos
cd "$testname"
arglist=""
runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=test_extend_border
cd "$testname"
arglist=""
runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=test_gauss2d
cd "$testname"
arglist=""
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=test_kg
cd "$testname"
arglist=""
runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=test_kxpos
cd "$testname"
arglist=""
runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=test_kypos
cd "$testname"
arglist=""
runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

tetstname=test_mirror_BCs
cd "$testname"
arglist=""
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=test_mpi_specifyrowscolumns
if test $usempi -eq 1
then
    cd "$testname"
    arglist="-p input/test_mpi_specifyrowscolumns"
    mpirunandecho 6 $testname $testname Debug/$testname $arglist
else
    echo "Skipping MPI-only test $testname"
fi
cd "$workspacedir"

testname=test_nearby_neighbor
cd "$testname"
arglist=""
runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=test_patch_head
cd "$testname"
arglist=""
runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=test_post_weights
cd "$testname"
arglist="-p input/test_post_weights.params"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=test_sign
cd "$testname"
arglist=""
runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=TriggerTest
arglist="-p input/TriggerTest.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=TransposeConnTest
cd "$testname"
arglist="" # parameter filename is in main()
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=TransposeHyPerConnTest
cd "$testname"
arglist="" # parameter filename is in main()
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=UnequalPatchSizeTest
cd "$testname"
arglist="-p input/UnequalPatchSizeTest.params"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

testname=UpdateFromCloneTest
cd "$testname"
arglist="-p input/updateFromCloneTest.params"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

# # Windowing was marked obsolete Dec 2, 2014
#testname=WindowSystemTest
#cd "$testname"
#arglist="-p input/postTest.params"
#runandecho $testname $testname Debug/$testname $arglist
#mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
#cd "$workspacedir"

testname=WriteSparseFileTest
cd "$testname"
arglist="-p input/WriteSparseFileTest.params -c checkpoints/Checkpoint6 --testall"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$workspacedir"

status=0
if test -n "$fails"
then
    echo "The following tests failed: $fails"
    status=1
else
    echo "All tests succeeded."
fi

if test -n "$dne"
then
    echo "The following tests do not exist: $dne"
    status=1
fi

exit $status
