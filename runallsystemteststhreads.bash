#! /usr/bin/env bash
#
# This script runs each systems test, suppressing all output but reporting
# whether the test passed or failed.  The list of system is hardcoded in the
# script; you need to add a new section if a new test is created.
#
# The script assumes that it is in a folder that contains all the systems
# tests project folders.  It can be run from any directory.
#
# When run without arguments, it assumes that the library and the tests
# were compiled with MPI.  It therefore runs system tests with both
# a single process and multiple processes.
#
# To turn off the MPI-specific tests, do "runalltests.bash --nompi"
# To set the mpirun command, do "runalltests.bash --mpirun=/path/to/mpirun"
# If mpirun is not set on the command line, it defaults to the result of
# searching the path for mpiexec, and then mpirun.

# Navigate to directory containing systems tests.
cd $(dirname "$0")

wd=$PWD

echo cd "$wd"

fails=""
dne=""

function runandecho() {
    testname=$1
    shift
    logfilebasename=$1
    shift
    if $* &> ${logfilebasename}_1.log
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
        echo $PV_MPIRUN -np $numprocs $*
        if $PV_MPIRUN -np $numprocs $* &> "${logfilename}"
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
arglist="-t 10 -p input/BasicSystemTest.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=AdjustAxonalArborsTest
arglist="-t 10 -p input/AdjustAxonalArborsTest.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=ArborSystemTest
arglist="-t 10 -p input/test_arbors.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=BinningLayerTest
arglist="-t 10 -p input/BinningLayerTest.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=CheckpointSystemTest
arglist="-t 10 "
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=CloneKernelConnTest
arglist="-t 10 -p input/CloneKernelConnTest.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=CloneVLayerTest
arglist="-t 10 "
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=DatastoreDelayTest
arglist="-t 10 -p input/DatastoreDelayTest.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=DelaysToFeaturesTest
arglist="-t 10 -p input/test_delays.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=FourByFourGenerativeTest
arglist="-t 10 "
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=FourByFourTopDownTest
arglist="-t 10 "
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=GenerativeConnTest
arglist="-t 10 "
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=GenericSystemTest
arglist="-t 10 -p input/GenericSystemTest.params -c checkpoints/Checkpoint6 --testall"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

#testname=GPUSystemTest
#cd "$testname"
#arglist="-d 0 -t 10 -p input/test_gpu.params"
#runandecho $testname $testname Debug/$testname $arglist
#mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
#cd "$wd"
echo "TODO: fix GPUSystemTest and maybe implement GPUs"

testname=ImageSystemTest
arglist="-t 10 -p input/multiframe_SystemTest.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=ImportParamsTest
arglist="-t 10 -p input/ImportParamsTest.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=InitWeightsTest
arglist="-t 10 -p input/test_initweights.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=KernelTest
logfilebasename=test_kernel
arglist="-t 10 -p input/test_kernel.params"
cd "$testname"
runandecho $testname $logfilebasename Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $logfilebasename Debug/$testname $arglist
cd "$wd"

testname=KernelTest
logfilebasename=test_kernel_normalizepost_shrunken
arglist="-t 10 -p input/test_kernel_normalizepost_shrunken.params"
cd "$testname"
runandecho $testname $logfilebasename Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $logfilebasename Debug/$testname $arglist
cd "$wd"

testname=LayerPhaseTest
arglist="-t 10 -p input/LayerPhaseTest.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=LayerRestartTest
arglist="-t 10 "
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=LIFTest
arglist="-t 10 -p input/LIFTest.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=MarginWidthTest
arglist="-t 10 -p input/MarginWidthTest.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=MatchingPursuitTest
arglist="-t 10 "
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=MLPTest
arglist="-t 10 "
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=MovieSystemTest
arglist="-t 10 -p input/MovieSystemTest.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=MPITest2
arglist="-t 10 -p input/MPI_test.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=NormalizeSystemTest
arglist="-t 10 -p input/NormalizeSystemTest.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=ParameterSweepTest
cd "$testname"
arglist="-t 10 -p input/ParameterSweepTest.params"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=PlasticConnTest
cd "$testname"
arglist="-t 10 -p input/PlasticConnTest.params"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=RandStateSystemTest
cd "$testname"
arglist="-t 10 -p input/RandStateSystemTest.params"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=ReadArborFileTest
cd "$testname"
arglist="-t 10 -p input/ReadArborFileTest.params"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=ReceiveFromPostTest
cd "$testname"
arglist="-t 10 -p input/postTest.params"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=ShrunkenPatchFlagTest
cd "$testname"
arglist="-t 10 -p input/ShrunkenPatchFlagTest.params" # parameter filename is in main()
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=ShrunkenPatchTest
cd "$testname"
arglist="-t 10 " # parameter filename is in main()
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=StochasticReleaseTest
cd "$testname"
arglist="-t 10 -p input/StochasticReleaseTest.params"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=test_border_activity
cd "$testname"
arglist="-t 10 "
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=test_cocirc
cd "$testname"
arglist="-t 10 -p input/test_cocirc.params"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=test_constant_input
cd "$testname"
arglist="-t 10 -p input/test_constant_input.params"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=test_delta
cd "$testname"
arglist="-t 10 "
runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=test_delta_pos
cd "$testname"
arglist="-t 10 "
runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=test_extend_border
cd "$testname"
arglist="-t 10 "
runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=test_gauss2d
cd "$testname"
arglist="-t 10 "
runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=test_kg
cd "$testname"
arglist="-t 10 "
runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=test_kxpos
cd "$testname"
arglist="-t 10 "
runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=test_kypos
cd "$testname"
arglist="-t 10 "
runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

tetstname=test_mirror_BCs
cd "$testname"
arglist="-t 10 "
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=test_mpi_specifyrowscolumns
if test $usempi -eq 1
then
    cd "$testname"
    arglist="-t 10 -p input/test_mpi_specifyrowscolumns"
    mpirunandecho 6 $testname $testname Debug/$testname $arglist
else
    echo "Skipping MPI-only test $testname"
fi
cd "$wd"

testname=test_nearby_neighbor
cd "$testname"
arglist="-t 10 "
runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=test_patch_head
cd "$testname"
arglist="-t 10 "
runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=test_post_weights
cd "$testname"
arglist="-t 10 -p input/test_post_weights.params"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=test_sign
cd "$testname"
arglist="-t 10 "
runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=TriggerTest
arglist="-t 10 -p input/TriggerTest.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=TransposeConnTest
cd "$testname"
arglist="-t 10 " # parameter filename is in main()
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=WindowSystemTest
cd "$testname"
arglist="-t 10 -p input/postTest.params"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

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
