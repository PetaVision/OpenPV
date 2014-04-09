#! /usr/bin/env bash
#
# This script runs each systems test and each unit test, suppressing all
# output but reporting whether the test passed or failed.  The system tests
# that get run are hardcoded in the script; you need to add a new section
# if a new test is created.
#
# The script assumes that it is in the PetaVision/scripts directory.
# It can be run from any directory.
#
# When run without arguments, it assumes that the library and the tests
# were compiled with MPI.  It therefore runs system tests with both
# a single process and multiple processes, and it runs unit tests executed
# by make runMPItests from within PetaVision/tests.
#
# To turn off the MPI-specific tests, do "runalltests.bash --nompi"
# To set the mpirun command, do "runalltests.bash --mpirun=/path/to/mpirun"
# If mpirun is not set on the command line,
# it defaults to "mpiexec-openmpi-mp" on Macs, and "mpirun" on Linux.

# Navigate to eclipse workspace directory.
if test "${0%/*}" != "$0"
then
    cd "${0%/*}"
fi
cd ../..
wd=$PWD # $wd is the eclipse workspace directory

echo cd $wd

fails=""

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

# Set default to use MPI, with run-command either
# mpiexec-openmpi-mp (Macs) or mpirun (Linux)
usempi=1
if test "$(uname)" = "Darwin"
then
    PV_MPIRUN=mpiexec-openmpi-mp
elif test "$(uname)" = "Linux"
then
    PV_MPIRUN=mpirun
fi

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
        echo "$0 error: mpirun command cannot be empty." > /dev/stderr
        exit 1
    fi
    function mpirunandecho() {
        numprocs=$1
        shift
        testname=$1
        shift
        logfilebasename=$1
        shift
        logfilename="${logfilebasename}_${numprocs}.log"
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
arglist="-p input/BasicSystemTest.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd $wd

testname=AdjustAxonalArborsTest
arglist="-p input/AdjustAxonalArborsTest.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd $wd

testname=ArborSystemTest
arglist="-p input/test_arbors.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd $wd

testname=BinningLayerTest
arglist="-p input/BinningLayerTest.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd $wd

testname=CheckpointSystemTest
arglist=""
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd $wd

testname=CloneKernelConnTest
arglist="-p input/CloneKernelConnTest.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd $wd

testname=CloneVLayerTest
arglist=""
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd $wd

testname=DatastoreDelayTest
arglist="-p input/DatastoreDelayTest.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd $wd

testname=DelaysToFeaturesTest
arglist="-p input/test_delays.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd $wd

testname=FourByFourGenerativeTest
arglist=""
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
cd $wd

testname=FourByFourTopDownTest
arglist=""
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
cd $wd

testname=GenerativeConnTest
arglist=""
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd $wd

testname=GenericSystemTest
arglist="-p input/GenericSystemTest.params -c checkpoints/Checkpoint6 --testall"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd $wd

#testname=GPUSystemTest
#cd "$testname"
#arglist="-d 0 -p input/test_gpu.params"
#runandecho $testname $testname Debug/$testname $arglist
#mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
#cd "$wd"
echo "TODO: fix GPUSystemTest and maybe implement GPUs"

testname=ImageSystemTest
arglist="-p input/multiframe_SystemTest.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd $wd

testname=ImportParamsTest
arglist="-p input/ImportParamsTest.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd $wd

testname=InitWeightsTest
arglist="-p input/test_initweights.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd $wd

testname=KernelTest
logfilebasename=test_kernel
arglist="-p input/test_kernel.params"
cd "$testname"
runandecho $testname $logfilebasename Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $logfilebasename Debug/$testname $arglist
cd $wd

testname=KernelTest
logfilebasename=test_kernel_normalizepost_shrunken
arglist="-p input/test_kernel_normalizepost_shrunken.params"
cd "$testname"
runandecho $testname $logfilebasename Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $logfilebasename Debug/$testname $arglist
cd $wd

testname=LayerPhaseTest
arglist="-p input/LayerPhaseTest.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd $wd

testname=LayerRestartTest
arglist=""
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd $wd

testname=LIFTest
arglist="-p input/LIFTest.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd $wd

testname=MarginWidthTest
arglist="-p input/MarginWidthTest.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd $wd

testname=MatchingPursuitTest
arglist=""
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd $wd

testname=MovieSystemTest
arglist="-p input/MovieSystemTest.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd $wd

testname=MPITest2
arglist="-p input/MPI_test.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd $wd

testname=NormalizeSystemTest
arglist="-p input/NormalizeSystemTest.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
cd $wd

testname=ParameterSweepTest
cd "$testname"
arglist="-p input/ParameterSweepTest.params"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$wd"


testname=PlasticConnTest
cd "$testname"
arglist="-p input/PlasticConnTest.params"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=RandStateSystemTest
cd "$testname"
arglist="-p input/RandStateSystemTest.params"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=ReadArborFileTest
cd "$testname"
arglist="-p input/ReadArborFileTest.params"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=ReceiveFromPostTest
cd "$testname"
arglist="-p input/postTest.params"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=ShrunkenPatchFlagTest
cd "$testname"
arglist="-p input/ShrunkenPatchFlagTest.params" # parameter filename is in main()
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=ShrunkenPatchTest
cd "$testname"
arglist="" # parameter filename is in main()
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$wd"


testname=StochasticReleaseTest
cd "$testname"
arglist="-p input/StochasticReleaseTest.params"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=test_border_activity
cd "$testname"
arglist=""
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd $wd

testname=test_cocirc
cd "$testname"
arglist="-p input/test_cocirc.params"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd $wd

testname=test_constant_input
cd "$testname"
arglist="-p input/test_constant_input.params"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd $wd

testname=test_delta
cd "$testname"
arglist=""
runandecho $testname $testname Debug/$testname $arglist
cd $wd

testname=test_delta_pos
cd "$testname"
arglist=""
runandecho $testname $testname Debug/$testname $arglist
cd $wd

testname=test_extend_border
cd "$testname"
arglist=""
runandecho $testname $testname Debug/$testname $arglist
cd $wd

testname=test_gauss2d
cd "$testname"
arglist=""
runandecho $testname $testname Debug/$testname $arglist
cd $wd

testname=test_kg
cd "$testname"
arglist=""
runandecho $testname $testname Debug/$testname $arglist
cd $wd

testname=test_kxpos
cd "$testname"
arglist=""
runandecho $testname $testname Debug/$testname $arglist
cd $wd

testname=test_kypos
cd "$testname"
arglist=""
runandecho $testname $testname Debug/$testname $arglist
cd $wd

tetstname=test_mirror_BCs
cd "$testname"
arglist=""
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd $wd

testname=test_mpi_specifyrowscolumns
if test $usempi -eq 1
then
    cd "$testname"
    arglist="-p input/test_mpi_specifyrowscolumns"
    mpirunandecho 6 $testname $testname Debug/$testname $arglist
else
    echo "Skipping MPI-only test $testname"
fi
cd $wd

testname=test_nearby_neighbor
cd "$testname"
arglist=""
runandecho $testname $testname Debug/$testname $arglist
cd $wd

testname=test_patch_head
cd "$testname"
arglist=""
runandecho $testname $testname Debug/$testname $arglist
cd $wd

testname=test_post_weights
cd "$testname"
arglist="-p input/test_post_weights.params"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd $wd

testname=test_sign
cd "$testname"
arglist=""
runandecho $testname $testname Debug/$testname $arglist
cd $wd

testname=TriggerTest
arglist="-p input/TriggerTest.params"
cd "$testname"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd $wd

testname=TransposeConnTest
cd "$testname"
arglist="" # parameter filename is in main()
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

testname=WindowSystemTest
cd "$testname"
arglist="-p input/postTest.params"
runandecho $testname $testname Debug/$testname $arglist
mpi_np2_np4_runandecho $testname $testname Debug/$testname $arglist
cd "$wd"

if test -n "$fails"
then
    echo "The following tests failed: $fails"
    exit 1
else
    echo "All tests succeeded."
fi
