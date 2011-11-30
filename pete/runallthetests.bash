#! /usr/bin/env bash
if test "${0%/*}" != "$0"
then
    cd "${0%/*}"
fi
cd ..
wd=$PWD

testname=ArborSystemTest
arglist="-p input/test_arbors.params"
cd "$testname"
if Debug/$testname $arglist 1>/dev/null 2>/dev/null
then
    echo "$testname passed"
else
    echo "$testname FAILED"
fi
if mpirun -np 4 Debug/$testname $arglist 1>/dev/null 2>/dev/null
then
    echo "$testname with four processes passed"
else
    echo "$testname with four processes FAILED"
fi
cd $wd

testname=DatastoreDelayTest
arglist="-p input/DatastoreDelayTest.params"
cd "$testname"
if Debug/$testname $arglist 1>/dev/null 2>/dev/null
then
    echo "$testname passed"
else
    echo "$testname FAILED"
fi
if mpirun -np 4 Debug/$testname $arglist 1>/dev/null 2>/dev/null
then
    echo "$testname with four processes passed"
else
    echo "$testname with four processes FAILED"
fi
cd $wd

testname=FourByFourGenerativeTest
arglist=""
cd "$testname"
if Debug/$testname $arglist 1>/dev/null 2>/dev/null
then
    echo "$testname passed"
else
    echo "$testname FAILED"
fi
cd $wd

testname=FourByFourTopDownTest
arglist=""
cd "$testname"
if Debug/$testname $arglist 1>/dev/null 2>/dev/null
then
    echo "$testname passed"
else
    echo "$testname FAILED"
fi
cd $wd

testname=GenerativeConnTest
arglist=""
cd "$testname"
if Debug/$testname $arglist 1>/dev/null 2>/dev/null
then
    echo "$testname passed"
else
    echo "$testname FAILED"
fi
if mpirun -np 4 Debug/$testname $arglist 1>/dev/null 2>/dev/null
then
    echo "$testname with four processes passed"
else
    echo "$testname with four processes FAILED"
fi
cd $wd

testname=InitWeightsTest
arglist="-p input/test_initweights.params"
cd "$testname"
if Debug/$testname $arglist 1>/dev/null 2>/dev/null
then
    echo "$testname passed"
else
    echo "$testname FAILED"
fi
if mpirun -np 4 Debug/$testname $arglist 1>/dev/null 2>/dev/null
then
    echo "$testname with four processes passed"
else
    echo "$testname with four processes FAILED"
fi
cd $wd

testname=KernelTest
arglist="-p input/test_kernel.params"
cd "$testname"
if Debug/$testname $arglist 1>/dev/null 2>/dev/null
then
    echo "$testname passed"
else
    echo "$testname FAILED"
fi
if mpirun -np 4 Debug/$testname $arglist 1>/dev/null 2>/dev/null
then
    echo "$testname with four processes passed"
else
    echo "$testname with four processes FAILED"
fi
cd $wd

testname=LayerRestartTest
cd "$testname"
if Debug/$testname 1>/dev/null 2>/dev/null
then
    echo "$testname passed"
else
    echo "$testname FAILED"
fi
if mpirun -np 4 Debug/$testname 1>/dev/null 2>/dev/null
then
    echo "$testname with four processes passed"
else
    echo "$testname with four processes FAILED"
fi
cd $wd

testname=MPITest2
arglist="-p input/MPI_test.params -n 100"
cd "$testname"
if Debug/$testname $arglist 1>/dev/null 2>/dev/null
then
    echo "$testname passed"
else
    echo "$testname FAILED"
fi
if mpirun -np 4 Debug/$testname $arglist 1>/dev/null 2>/dev/null
then
    echo "$testname with four processes passed"
else
    echo "$testname with four processes FAILED"
fi
cd $wd

testname=PlasticConnTest
cd "$testname"
arglist="-p input/PlasticConnTest.params"
if Debug/$testname $arglist 1>/dev/null 2>/dev/null
then
    echo "$testname passed"
else
    echo "$testname FAILED"
fi
if mpirun -np 4 Debug/$testname $arglist 1>/dev/null 2>/dev/null
then
    echo "$testname with four processes passed"
else
    echo "$testname with four processes FAILED"
fi
cd "$wd"

cd "PetaVision/tests"
make runtests 2>/dev/null | egrep 'passed|failed'
cd $wd
