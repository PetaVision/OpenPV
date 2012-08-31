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

# Navigate to eclipse workspace directory.
if test "${0%/*}" != "$0"
then
    cd "${0%/*}"
fi
cd ../..
wd=$PWD # $wd is the eclipse workspace directory

echo $wd

function runandecho() {
    testname=$1
    shift
    if $* 1> /dev/null 2>/dev/null
    then
        echo "$testname passed"
    else
        echo "$testname FAILED"
    fi
}

# Check for --nompi option.
if test ${1:-usempi} = "--nompi"
then
    usempi=0
    function mpirunandecho() {
        false
    }
else
    usempi=1
    function mpirunandecho() {
        testname=$1
        shift
        if openmpirun -np 4 $* 1> /dev/null 2>/dev/null
        then
            echo "$testname with four processes passed"
        else
            echo "$testname with four processes FAILED"
        fi
    }
fi

testname=BasicSystemTest
arglist="-p input/BasicSystemTest.params"
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

testname=CheckpointSystemTest
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

testname=GPUSystemTest
cd "$testname"
arglist="-d 0 -p input/test_gpu.params"
runandecho $testname Debug/$testname $arglist
mpirunandecho $testname Debug/$testname $arglist
cd "$wd"

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

testname=MPITest2
arglist="-p input/MPI_test.params -n 100"
cd "$testname"
runandecho $testname Debug/$testname $arglist
mpirunandecho $testname Debug/$testname $arglist
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

testname=ReadArborFileTest
cd "$testname"
arglist="-p input/ReadArborFileTest.params"
runandecho $testname Debug/$testname $arglist
mpirunandecho $testname Debug/$testname $arglist
cd "$wd"

cd "./PetaVision/tests"
make runtests 2>/dev/null | egrep 'passed|FAILED'
if test $usempi -eq 1
then
    make runMPItests 2>/dev/null | egrep 'passed|FAILED'
fi
cd $wd
