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

# Define mpirun command.  On Macs, macports installs mpi commands as openmpi*
# On neuro/anterior/etc.  Linux installs them as mpi*
if test "$(uname)" = "Darwin"
then
    PV_MPIRUN=openmpirun
elif test "$(uname)" = "Linux"
then
    PV_MPIRUN=mpirun
fi

# Navigate to directory containing systems tests.
cd $(dirname "$0")

wd=$PWD

echo cd $wd

fails=""
dne=""

# # Open MPI 1.7 on Macs has a bug in MPI_Finalize that causes a crash on
# # exit if using GDAL if you run outside of openmpirun.  As a workaround,
# # runandecho is defined differently depending on whether --nompi is used.
#function runandecho() {
#    testname=$1
#    shift
#    if $* &> ${testname}_1.log ## 1> /dev/null 2>/dev/null
#    then
#        echo "$testname passed"
#    else
#        echo "$testname FAILED"
#        fails="$fails $testname"
#    fi
#}

# Check for --nompi option.
if test ${1:---usempi} = "--nompi"
then
    usempi=0
    function runandecho() {
        testname=$1
        shift
        if $* &> ${testname}_1.log ## 1> /dev/null 2>/dev/null
        then
            echo "$testname passed"
        else
            echo "$testname FAILED"
            fails="$fails $testname"
        fi
    }
    function mpirunandecho() {
        false
    }
else
    usempi=1
    # On the Mac, we need to do call within openmpirun even if using just
    # one processor.  For convenience, we define runandecho() accordingly
    # even if using Linux.
    function runandecho() {
        testname=$1
        shift
        if $PV_MPIRUN -np 1 $* &> ${testname}_1.log ## 1> /dev/null 2>/dev/null
        then
            echo "$testname passed"
        else
            echo "$testname FAILED"
            fails="$fails $testname"
        fi
    }
    function mpirunandecho() {
        testname=$1
        shift
        if $PV_MPIRUN -np 2 $* &> ${testname}_2.log ## 1> /dev/null 2>/dev/null
        then
            echo "$testname with two processes passed"
        else
            echo "$testname with two processes FAILED"
            fails="$fails $testname(2 procs)"
        fi
        if $PV_MPIRUN -np 4 $* &> ${testname}_4.log ## 1> /dev/null 2>/dev/null
        then
            echo "$testname with four processes passed"
        else
            echo "$testname with four processes FAILED"
            fails="$fails $testname(4 procs)"
        fi

    }
fi


testname=BasicSystemTest
arglist="-p input/BasicSystemTest.params"
if cd "$testname"
then
    runandecho $testname Debug/$testname $arglist
    mpirunandecho $testname Debug/$testname $arglist
    cd $wd
else
    echo "$testname does not exist"
    dne="$dne $testname"
fi

testname=AdjustAxonalArborsTest
arglist="-p input/AdjustAxonalArborsTest.params"
cd "$testname"
runandecho $testname Debug/$testname $arglist
mpirunandecho $testname Debug/$testname $arglist
cd $wd


testname=ArborSystemTest
arglist="-p input/test_arbors.params"
cd "$testname"
runandecho $testname Debug/$testname $arglist
mpirunandecho $testname Debug/$testname $arglist
cd $wd

testname=BinningLayerTest
arglist="-p input/BinningLayerTest.params"
cd "$testname"
runandecho $testname Debug/$testname $arglist
mpirunandecho $testname Debug/$testname $arglist
cd $wd

testname=CheckpointSystemTest
arglist=""
cd "$testname"
runandecho $testname Debug/$testname $arglist
mpirunandecho $testname Debug/$testname $arglist
cd $wd

testname=CloneKernelConnTest
arglist="-p input/CloneKernelConnTest.params"
cd "$testname"
runandecho $testname Debug/$testname $arglist
mpirunandecho $testname Debug/$testname $arglist
cd $wd

testname=CloneVLayerTest
arglist=""
cd "$testname"
runandecho $testname Debug/$testname $arglist
mpirunandecho $testname Debug/$testname $arglist
cd $wd

testname=DatastoreDelayTest
arglist="-p input/DatastoreDelayTest.params"
cd "$testname"
runandecho $testname Debug/$testname $arglist
mpirunandecho $testname Debug/$testname $arglist
cd $wd

testname=DelaysToFeaturesTest
arglist="-p input/test_delays.params"
cd "$testname"
runandecho $testname Debug/$testname $arglist
mpirunandecho $testname Debug/$testname $arglist
cd $wd

testname=FourByFourGenerativeTest
arglist=""
cd "$testname"
runandecho $testname Debug/$testname $arglist
cd $wd

testname=FourByFourTopDownTest
arglist=""
cd "$testname"
runandecho $testname Debug/$testname $arglist
cd $wd

testname=GenerativeConnTest
arglist=""
cd "$testname"
runandecho $testname Debug/$testname $arglist
mpirunandecho $testname Debug/$testname $arglist
cd $wd

#testname=GPUSystemTest
#cd "$testname"
#arglist="-d 0 -p input/test_gpu.params"
#runandecho $testname Debug/$testname $arglist
#mpirunandecho $testname Debug/$testname $arglist
#cd "$wd"
echo "TODO: fix GPUSystemTest and maybe implement GPUs"

testname=ImageSystemTest
arglist="-p input/multiframe_SystemTest.params"
cd "$testname"
runandecho $testname Debug/$testname $arglist
mpirunandecho $testname Debug/$testname $arglist
cd $wd

testname=ImportParamsTest
arglist="-p input/ImportParamsTest.params"
cd "$testname"
runandecho $testname Debug/$testname $arglist
mpirunandecho $testname Debug/$testname $arglist
cd $wd

testname=InitWeightsTest
arglist="-p input/test_initweights.params"
cd "$testname"
runandecho $testname Debug/$testname $arglist
mpirunandecho $testname Debug/$testname $arglist
cd $wd

testname=KernelTest
arglist="-p input/test_kernel.params"
cd "$testname"
runandecho $testname Debug/$testname $arglist
mpirunandecho $testname Debug/$testname $arglist
cd $wd

testname=KernelTest
arglist="-p input/test_kernel_normalizepost_shrunken.params"
cd "$testname"
runandecho $testname Debug/$testname $arglist
mpirunandecho $testname Debug/$testname $arglist
cd $wd

testname=LayerPhaseTest
arglist="-p input/LayerPhaseTest.params"
cd "$testname"
runandecho $testname Debug/$testname $arglist
mpirunandecho $testname Debug/$testname $arglist
cd $wd

testname=LayerRestartTest
arglist=""
cd "$testname"
runandecho $testname Debug/$testname $arglist
mpirunandecho $testname Debug/$testname $arglist
cd $wd

testname=LIFTest
arglist="-p input/LIFTest.params"
cd "$testname"
runandecho $testname Debug/$testname $arglist
mpirunandecho $testname Debug/$testname $arglist
cd $wd

testname=MarginWidthTest
arglist="-p input/MarginWidthTest.params"
cd "$testname"
runandecho $testname Debug/$testname $arglist
mpirunandecho $testname Debug/$testname $arglist
cd $wd

testname=MatchingPursuitTest
arglist=""
cd "$testname"
runandecho $testname Debug/$testname $arglist
mpirunandecho $testname Debug/$testname $arglist
cd $wd

testname=MovieSystemTest
arglist="-p input/MovieSystemTest.params"
cd "$testname"
runandecho $testname Debug/$testname $arglist
mpirunandecho $testname Debug/$testname $arglist
cd $wd

testname=MPITest2
arglist="-p input/MPI_test.params"
cd "$testname"
runandecho $testname Debug/$testname $arglist
mpirunandecho $testname Debug/$testname $arglist
cd $wd

testname=NormalizeSystemTest
arglist="-p input/NormalizeSystemTest.params"
cd "$testname"
runandecho $testname Debug/$testname $arglist
cd $wd

testname=ParameterSweepTest
cd "$testname"
arglist="-p input/ParameterSweepTest.params"
runandecho $testname Debug/$testname $arglist
mpirunandecho $testname Debug/$testname $arglist
cd "$wd"


testname=PlasticConnTest
cd "$testname"
arglist="-p input/PlasticConnTest.params"
runandecho $testname Debug/$testname $arglist
mpirunandecho $testname Debug/$testname $arglist
cd "$wd"

testname=RandStateSystemTest
cd "$testname"
arglist="-p input/RandStateSystemTest.params"
runandecho $testname Debug/$testname $arglist
mpirunandecho $testname Debug/$testname $arglist
cd "$wd"

testname=ReadArborFileTest
cd "$testname"
arglist="-p input/ReadArborFileTest.params"
runandecho $testname Debug/$testname $arglist
mpirunandecho $testname Debug/$testname $arglist
cd "$wd"

testname=ReceiveFromPostTest
cd "$testname"
arglist="-p input/postTest.params"
runandecho $testname Debug/$testname $arglist
mpirunandecho $testname Debug/$testname $arglist
cd "$wd"

testname=ShrunkenPatchTest
cd "$testname"
arglist="" # parameter filename is in main()
runandecho $testname Debug/$testname $arglist
mpirunandecho $testname Debug/$testname $arglist
cd "$wd"

testname=ShrunkenPatchFlagTest
cd "$testname"
arglist="-p input/ShrunkenPatchFlagTest.params"
runandecho $testname Debug/$testname $arglist
mpirunandecho $testname Debug/$testname $arglist
cd "$wd"

testname=StochasticReleaseTest
cd "$testname"
arglist="-p StochasticReleaseTest.params"
runandecho $testname Debug/$testname $arglist
mpirunandecho $testname Debug/$testname $arglist
cd "$wd"

testname=test_border_activity
cd "$testname"
arglist=""
runandecho $testname Debug/$testname $arglist
mpirunandecho $testname Debug/$testname $arglist
cd "$wd"

testname=test_cocirc
cd "$testname"
arglist="-p input/test_cocirc.params"
runandecho $testname Debug/$testname $arglist
mpirunandecho $testname Debug/$testname $arglist
cd "$wd"

testname="test_gauss2d"
cd "$testname"
arglist=""
runandecho $testname Debug/$testname $arglist
mpirunandecho $testname Debug/$testname $arglist
cd "$wd"

testname=TransposeConnTest
cd "$testname"
arglist=""
runandecho $testname Debug/$testname $arglist
mpirunandecho $testname Debug/$testname $arglist
cd "$wd"

testname=TriggerTest
arglist="-p input/TriggerTest.params"
cd "$testname"
runandecho $testname Debug/$testname $arglist
mpirunandecho $testname Debug/$testname $arglist
cd $wd

testname=WindowSystemTest
cd "$testname"
arglist="-p input/postTest.params" # parameter filename is in main()
runandecho $testname Debug/$testname $arglist
mpirunandecho $testname Debug/$testname $arglist
cd "$wd"

# # Unit tests
cd UnitTests

testname="test_kg"
if ./$testname &> ${testname}_1.log
then
    echo "$testname passed."
else
    echo "$testname FAILED."
    fails="$fails $testname"
fi

testname="test_kxpos"
if ./$testname &> ${testname}_1.log
then
    echo "$testname passed."
else
    echo "$testname FAILED."
    fails="$fails $testname"
fi

testname="test_kypos"
if ./$testname &> ${testname}_1.log
then
    echo "$testname passed."
else
    echo "$testname FAILED."
    fails="$fails $testname"
fi

testname="test_nearby_neighbor"
if ./$testname &> ${testname}_1.log
then
    echo "$testname passed."
else
    echo "$testname FAILED."
    fails="$fails $testname"
fi

testname="test_patch_head"
if ./$testname &> ${testname}_1.log
then
    echo "$testname passed."
else
    echo "$testname FAILED."
    fails="$fails $testname"
fi

testname="test_post_weights"
if ./$testname -p input/test_post_weights.params &> ${testname}_1.log
then
    echo "$testname passed."
else
    echo "$testname FAILED."
    fails="$fails $testname"
fi

testname="test_delta_pos"
if ./$testname &> ${testname}_1.log
then
    echo "$testname passed."
else
    echo "$testname FAILED."
    fails="$fails $testname"
fi

testname="test_extend_border"
if ./$testname &> ${testname}_1.log
then
    echo "$testname passed."
else
    echo "$testname FAILED."
    fails="$fails $testname"
fi

testname="test_patch_head"
if ./$testname &> ${testname}_1.log
then
    echo "$testname passed."
else
    echo "$testname FAILED."
    fails="$fails $testname"
fi

testname="test_delta"
if ./$testname &> ${testname}_1.log
then
    echo "$testname passed."
else
    echo "$testname FAILED."
    fails="$fails $testname"
fi

testname="test_sign"
if ./$testname &> ${testname}_1.log
then
    echo "$testname passed."
else
    echo "$testname FAILED."
    fails="$fails $testname"
fi

testname="test_mirror_BCs"
if $PV_MPIRUN -np 1 ./$testname &> ${testname}_1.log
then
    echo "$testname passed."
else
    echo "$testname FAILED."
    fails="$fails $testname"
fi

# MPI unit tests

if test $usempi -ne 0
then
    testname="test_mpi_specifyrowscolumns"
    if $PV_MPIRUN -np 6 ./$testname -p input/test_mpi_specifyrowscolumns.params &> ${testname}_6.log
    then
        echo "$testname passed."
    else
        echo "$testname FAILED."
        fails="$fails $testname"
    fi
fi

cd $wd

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

if test status != 0; then exit 1; fi
