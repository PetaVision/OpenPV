#! /usr/bin/env bash

# If called from a directory other than PVSystemsTests, change directory to PVSystemsTests
if test "${0%/*}" != "$0"
then
    cd "${0%/*}"
fi
# We should now be in the PVSystemsTests directory
wd=$PWD

testname=BasicSystemTest
params="input/$testname.params"
outdir="output"
outparams="$outdir/pv.params"
args="-p $params"
echo ==== $testname ===
cd $wd/$testname
rm -r output
openmpirun -np 1 Debug/$testname $args
mv $params{,.bak}
cp -p $outparams $params
openmpirun -np 1 Debug/$testname $args 2>&1 > ${testname}_ioparams.log ||
    echo "$testname FAILED" >> ${testname}_ioparams.log
mv $params{.bak,}

testname=AdjustAxonalArborsTest
params="input/$testname.params"
outdir="output"
outparams="$outdir/pv.params"
args="-p $params"
echo ==== $testname ===
cd $wd/$testname
rm -r output
openmpirun -np 1 Debug/$testname $args
mv $params{,.bak}
cp -p $outparams $params
openmpirun -np 1 Debug/$testname $args 2>&1 > ${testname}_ioparams.log ||
    echo "$testname FAILED" >> ${testname}_ioparams.log
mv $params{.bak,}

testname=ArborSystemTest
params="input/test_arbors.params"
outdir="output"
outparams="$outdir/pv.params"
args="-p $params"
echo ==== $testname ===
cd $wd/$testname
rm -r output
openmpirun -np 1 Debug/$testname $args
mv $params{,.bak}
cp -p $outparams $params
openmpirun -np 1 Debug/$testname $args 2>&1 > ${testname}_ioparams.log ||
    echo "$testname FAILED" >> ${testname}_ioparams.log
mv $params{.bak,}

testname=BinningLayerTest
params="input/$testname.params"
outdir="output"
outparams="$outdir/pv.params"
args="-p $params"
echo ==== $testname ===
cd $wd/$testname
rm -r output
openmpirun -np 1 Debug/$testname $args
mv $params{,.bak}
cp -p $outparams $params
openmpirun -np 1 Debug/$testname $args 2>&1 > ${testname}_ioparams.log ||
    echo "$testname FAILED" >> ${testname}_ioparams.log
mv $params{.bak,}

testname=CheckpointSystemTest
echo ==== $testname ====
cd $wd/$testname
rm -r output checkpoints1 checkpoints2
openmpirun -np 1 Debug/$testname
mv input/CheckpointParameters1.params{,.bak}
mv input/CheckpointParameters2.params{,.bak}
cp -p output/pv1.params input/CheckpointParameters1.params
cp -p output/pv2.params input/CheckpointParameters2.params
openmpirun -np 1 Debug/$testname 2>&1 > ${testname}_ioparams.log ||
    echo "$testname FAILED" >> ${testname}_ioparams.log
mv input/CheckpointParameters1.params{.bak,}
mv input/CheckpointParameters2.params{.bak,}

testname=CloneKernelConnTest
params="input/$testname.params"
outdir="output"
outparams="$outdir/pv.params"
args="-p $params"
echo ==== $testname ===
cd $wd/$testname
rm -r output
openmpirun -np 1 Debug/$testname $args
mv $params{,.bak}
cp -p $outparams $params
openmpirun -np 1 Debug/$testname $args 2>&1 > ${testname}_ioparams.log ||
    echo "$testname FAILED" >> ${testname}_ioparams.log
mv $params{.bak,}

testname=CloneVLayerTest
params="input/$testname.params"
outdir="output"
outparams="$outdir/pv.params"
args="-p $params"
echo ==== $testname ===
cd $wd/$testname
rm -r output
openmpirun -np 1 Debug/$testname $args
mv $params{,.bak}
cp -p $outparams $params
openmpirun -np 1 Debug/$testname $args 2>&1 > ${testname}_ioparams.log ||
    echo "$testname FAILED" >> ${testname}_ioparams.log
mv $params{.bak,}

testname=DatastoreDelayTest
params="input/$testname.params"
outdir="output"
outparams="$outdir/pv.params"
args="-p $params"
echo ==== $testname ====
cd $wd/$testname
rm -r $outdir
openmpirun -np 1 Debug/$testname $args
mv $params{,.bak}
cp -p $outparams $params
openmpirun -np 1 Debug/$testname $args 2>&1 > ${testname}_ioparams.log ||
    echo "$testname FAILED" >> ${testname}_ioparams.log
mv $params{.bak,}

testname=DelaysToFeaturesTest
params="input/test_delays.params"
outdir="output"
outparams="$outdir/pv.params"
args="-p $params"
echo ==== $testname ====
cd $wd/$testname
rm -r $outdir
openmpirun -np 1 Debug/$testname $args
mv $params{,.bak}
cp -p $outparams $params
openmpirun -np 1 Debug/$testname $args 2>&1 > ${testname}_ioparams.log ||
    echo "$testname FAILED" >> ${testname}_ioparams.log
mv $params{.bak,}

testname=FourByFourGenerativeTest
params="input/${testname}.params"
outdir="output"
outparams="$outdir/pv.params"
args="-p $params"
echo ==== $testname ====
cd $wd/$testname
rm -r $outdir
openmpirun -np 1 Debug/$testname $args
mv $params{,.bak}
cp -p $outparams $params
openmpirun -np 1 Debug/$testname $args 2>&1 > ${testname}_ioparams.log ||
    echo "$testname FAILED" >> ${testname}_ioparams.log
mv $params{.bak,}

testname=FourByFourTopDownTest
params="input/${testname}.params"
outdir="output"
outparams="$outdir/pv.params"
args="-p $params"
echo ==== $testname ====
cd $wd/$testname
rm -r $outdir
openmpirun -np 1 Debug/$testname $args
mv $params{,.bak}
cp -p $outparams $params
openmpirun -np 1 Debug/$testname $args 2>&1 > ${testname}_ioparams.log ||
    echo "$testname FAILED" >> ${testname}_ioparams.log
mv $params{.bak,}

testname=GenerativeConnTest
echo ==== $testname ====
cd $wd/$testname
rm -r output-MirrorBC*
openmpirun -np 1 Debug/$testname
mv input/GenerativeConnTest-MirrorBCOff.params{,.bak}
mv input/GenerativeConnTest-MirrorBCOn.params{,.bak}
cp -p output-MirrorBCOn/pv.params input/GenerativeConnTest-MirrorBCOn.params
cp -p output-MirrorBCOn/pv.params input/GenerativeConnTest-MirrorBCOff.params
openmpirun -np 1 Debug/$testname 2>&1 > ${testname}_ioparams.log ||
    echo "$testname FAILED" >> ${testname}_ioparams.log
mv input/GenerativeConnTest-MirrorBCOff.params{.bak,}
mv input/GenerativeConnTest-MirrorBCOn.params{.bak,}

testname=ImageSystemTest
params="input/multiframe_SystemTest.params"
outdir="output"
outparams="$outdir/pv.params"
args="-p $params"
echo ==== $testname ====
cd $wd/$testname
rm -r $outdir
openmpirun -np 1 Debug/$testname $args
mv $params{,.bak}
cp -p $outparams $params
openmpirun -np 1 Debug/$testname $args 2>&1 > ${testname}_ioparams.log ||
    echo "$testname FAILED" >> ${testname}_ioparams.log
mv $params{.bak,}

testname=ImportParamsTest
params="input/${testname}.params"
outdir="output"
outparams="$outdir/pv.params"
args="-p $params"
echo ==== $testname ====
cd $wd/$testname
rm -r $outdir
openmpirun -np 1 Debug/$testname $args
mv $params{,.bak}
cp -p $outparams $params
openmpirun -np 1 Debug/$testname $args 2>&1 > ${testname}_ioparams.log ||
    echo "$testname FAILED" >> ${testname}_ioparams.log
mv $params{.bak,}

testname=InitWeightsTest
params="input/test_initweights.params"
outdir="output"
outparams="$outdir/pv.params"
args="-p $params"
echo ==== $testname ====
cd $wd/$testname
rm -r $outdir
openmpirun -np 1 Debug/$testname $args
mv $params{,.bak}
cp -p $outparams $params
openmpirun -np 1 Debug/$testname $args 2>&1 > ${testname}_ioparams.log ||
    echo "$testname FAILED" >> ${testname}_ioparams.log
mv $params{.bak,}

testname=KernelTest
params="input/test_kernel.params"
outdir="output"
outparams="$outdir/pv.params"
args="-p $params"
echo ==== $testname/test_kernel ====
cd $wd/$testname
rm -r $outdir
openmpirun -np 1 Debug/$testname $args
mv $params{,.bak}
cp -p $outparams $params
openmpirun -np 1 Debug/$testname $args 2>&1 > test_kernel_ioparams.log ||
    echo "$testname FAILED" >> ${testname}_ioparams.log
mv $params{.bak,}

testname=KernelTest
params="input/test_kernel_normalizepost_shrunken.params"
outdir="output"
outparams="$outdir/pv.params"
args="-p $params"
echo ==== $testname/test_kernel_normalizepost_shrunken ====
cd $wd/$testname
rm -r $outdir
openmpirun -np 1 Debug/$testname $args
mv $params{,.bak}
cp -p $outparams $params
openmpirun -np 1 Debug/$testname $args 2>&1 > test_kernel_normalizepost_shrunken_ioparams.log ||
    echo "$testname FAILED" >> ${testname}_ioparams.log
mv $params{.bak,}

testname=LayerPhaseTest
params="input/${testname}.params"
outdir="output"
outparams="$outdir/pv.params"
args="-p $params"
echo ==== $testname ====
cd $wd/$testname
rm -r $outdir
openmpirun -np 1 Debug/$testname $args
mv $params{,.bak}
cp -p $outparams $params
openmpirun -np 1 Debug/$testname $args 2>&1 > ${testname}_ioparams.log ||
    echo "$testname FAILED" >> ${testname}_ioparams.log
mv $params{.bak,}

testname=LayerRestartTest
echo ==== $testname ====
cd $wd/$testname
rm -r $outdir
openmpirun -np 1 Debug/$testname
for k in Write Check Read
do
    mv input/LayerRestartTest-$k.params{,.bak}
    cp -p output/pv-$k.params input/LayerRestartTest-$k.params
done
openmpirun -np 1 Debug/$testname 2>&1 > ${testname}_ioparams.log
for k in Write Check Read
do
    mv input/LayerRestartTest-$k.params{.bak,}
done

testname=LIFTest
params="input/${testname}.params"
outdir="output"
outparams="$outdir/pv.params"
args="-p $params"
echo ==== $testname ====
cd $wd/$testname
rm -r $outdir
openmpirun -np 1 Debug/$testname $args
mv $params{,.bak}
cp -p $outparams $params
openmpirun -np 1 Debug/$testname $args 2>&1 > ${testname}_ioparams.log ||
    echo "$testname FAILED" >> ${testname}_ioparams.log
mv $params{.bak,}

testname=MarginWidthTest
params="input/${testname}.params"
outdir="output"
outparams="$outdir/pv.params"
args="-p $params"
echo ==== $testname ====
cd $wd/$testname
rm -r $outdir
openmpirun -np 1 Debug/$testname $args
mv $params{,.bak}
cp -p $outparams $params
openmpirun -np 1 Debug/$testname $args 2>&1 > ${testname}_ioparams.log ||
    echo "$testname FAILED" >> ${testname}_ioparams.log
mv $params{.bak,}

testname=MatchingPursuitTest
params="input/${testname}.params"
outdir="output"
outparams="$outdir/pv.params"
args="-p $params"
echo ==== $testname ====
cd $wd/$testname
rm -r $outdir
openmpirun -np 1 Debug/$testname $args
mv $params{,.bak}
cp -p $outparams $params
openmpirun -np 1 Debug/$testname $args 2>&1 > ${testname}_ioparams.log ||
    echo "$testname FAILED" >> ${testname}_ioparams.log
mv $params{.bak,}

testname=MovieSystemTest
params="input/${testname}.params"
outdir="output"
outparams="$outdir/pv.params"
args="-p $params"
echo ==== $testname ====
cd $wd/$testname
rm -r $outdir
openmpirun -np 1 Debug/$testname $args
mv $params{,.bak}
cp -p $outparams $params
openmpirun -np 1 Debug/$testname $args 2>&1 > ${testname}_ioparams.log ||
    echo "$testname FAILED" >> ${testname}_ioparams.log
mv $params{.bak,}

testname=MPITest2
params="input/MPI_test.params"
outdir="output"
outparams="$outdir/pv.params"
args="-p $params"
echo ==== $testname ====
cd $wd/$testname
rm -r $outdir
openmpirun -np 1 Debug/$testname $args
mv $params{,.bak}
cp -p $outparams $params
openmpirun -np 1 Debug/$testname $args 2>&1 > ${testname}_ioparams.log ||
    echo "$testname FAILED" >> ${testname}_ioparams.log
mv $params{.bak,}

testname=NormalizeSystemTest
params="input/$testname.params"
outdir="output"
outparams="$outdir/pv.params"
args="-p $params"
echo ==== $testname ====
cd $wd/$testname
rm -r $outdir
openmpirun -np 1 Debug/$testname $args
mv $params{,.bak}
cp -p $outparams $params
openmpirun -np 1 Debug/$testname $args 2>&1 > ${testname}_ioparams.log ||
    echo "$testname FAILED" >> ${testname}_ioparams.log
mv $params{.bak,}

testname=ParameterSweepTest
params="input/$testname.params"
args="-p $params"
echo ==== $testname ====
cd $wd/$testname
rm -r output{1,3,5,7}
openmpirun -np 1 Debug/$testname $args
for k in 1 3 5 7
do
    cp -p output$k/pv.params input/pv$k.params
    openmpirun -np 1 Debug/$testname -p input/pv$k.params 2>&1 > ${testname}_ioparams.log ||
        echo "$testname FAILED" >> ${testname}_ioparams.log
    rm input/pv$k.params
done

testname=PlasticConnTest
params="input/$testname.params"
outdir="output"
outparams="$outdir/pv.params"
args="-p $params"
echo ==== $testname ====
cd $wd/$testname
rm -r $outdir
openmpirun -np 1 Debug/$testname $args
mv $params{,.bak}
cp -p $outparams $params
openmpirun -np 1 Debug/$testname $args 2>&1 > ${testname}_ioparams.log ||
    echo "$testname FAILED" >> ${testname}_ioparams.log
mv $params{.bak,}

testname=RandStateSystemTest
echo ==== $testname ====
cd $wd/$testname
rm -r output{1,2}
openmpirun -np 1 Debug/$testname
mv input/RandStateSystemTest1.params{,.bak}
mv input/RandStateSystemTest2{,.bak}
cp -p output1/pv.params input/RandStateSystemTest1.params
cp -p output2/pv.params input/RandStateSystemTest2.params
openmpirun -np 1 Debug/$testname 2>&1 > ${testname}_ioparams.log ||
    echo "$testname FAILED" >> ${testname}_ioparams.log
mv input/RandStateSystemTest1.params{.bak,}
mv input/RandStateSystemTest2{.bak,}

testname=ReadArborFileTest
params="input/$testname.params"
outdir="output"
outparams="$outdir/pv.params"
args="-p $params"
echo ==== $testname ====
cd $wd/$testname
rm -r $outdir
openmpirun -np 1 Debug/$testname $args
mv $params{,.bak}
cp -p $outparams $params
openmpirun -np 1 Debug/$testname $args 2>&1 > ${testname}_ioparams.log ||
    echo "$testname FAILED" >> ${testname}_ioparams.log
mv $params{.bak,}

testname=ReceiveFromPostTest
params="input/postTest.params"
outdir="output"
outparams="$outdir/pv.params"
args="-p $params"
echo ==== $testname ====
cd $wd/$testname
rm -r $outdir
openmpirun -np 1 Debug/$testname $args
mv $params{,.bak}
cp -p $outparams $params
openmpirun -np 1 Debug/$testname $args 2>&1 > ${testname}_ioparams.log ||
    echo "$testname FAILED" >> ${testname}_ioparams.log
mv $params{.bak,}

testname=ShrunkenPatchFlagTest
params="input/$testname.params"
outdir="output"
outparams="$outdir/pv.params"
args="-p $params"
echo ==== $testname ====
cd $wd/$testname
rm -r $outdir
openmpirun -np 1 Debug/$testname $args
mv $params{,.bak}
cp -p $outparams $params
openmpirun -np 1 Debug/$testname $args 2>&1 > ${testname}_ioparams.log ||
    echo "$testname FAILED" >> ${testname}_ioparams.log
mv $params{.bak,}

testname=ShrunkenPatchTest
params="input/$testname.params"
outdir="output"
outparams="$outdir/pv.params"
args="-p $params"
echo ==== $testname ====
cd $wd/$testname
rm -r $outdir
openmpirun -np 1 Debug/$testname $args
mv $params{,.bak}
cp -p $outparams $params
openmpirun -np 1 Debug/$testname $args 2>&1 > ${testname}_ioparams.log ||
    echo "$testname FAILED" >> ${testname}_ioparams.log
mv $params{.bak,}

testname=StochasticReleaseTest
params="input/$testname.params"
outdir="output"
outparams="$outdir/pv.params"
args="-p $params"
echo ==== $testname ====
cd $wd/$testname
rm -r $outdir
openmpirun -np 1 Debug/$testname $args
mv $params{,.bak}
cp -p $outparams $params
openmpirun -np 1 Debug/$testname $args 2>&1 > ${testname}_ioparams.log ||
    echo "$testname FAILED" >> ${testname}_ioparams.log
mv $params{.bak,}

testname=TriggerTest
params="input/$testname.params"
outdir="output"
outparams="$outdir/pv.params"
args="-p $params"
echo ==== $testname ====
cd $wd/$testname
rm -r $outdir
openmpirun -np 1 Debug/$testname $args
mv $params{,.bak}
cp -p $outparams $params
openmpirun -np 1 Debug/$testname $args 2>&1 > ${testname}_ioparams.log ||
    echo "$testname FAILED" >> ${testname}_ioparams.log
mv $params{.bak,}

testname=TransposeConnTest
params="input/$testname.params"
outdir="output"
outparams="$outdir/pv.params"
args="-p $params"
echo ==== $testname ====
cd $wd/$testname
rm -r $outdir
openmpirun -np 1 Debug/$testname $args
mv $params{,.bak}
cp -p $outparams $params
openmpirun -np 1 Debug/$testname $args 2>&1 > ${testname}_ioparams.log ||
    echo "$testname FAILED" >> ${testname}_ioparams.log
mv $params{.bak,}

# # Windowing was marked obsolete Dec 2, 2014
#testname=WindowSystemTest
#params="input/postTest.params"
#outdir="output"
#outparams="$outdir/pv.params"
#args="-p $params"
#echo ==== $testname ====
#cd $wd/$testname
#rm -r $outdir
#openmpirun -np 1 Debug/$testname $args
#mv $params{,.bak}
#cp -p $outparams $params
#openmpirun -np 1 Debug/$testname $args 2>&1 > ${testname}_ioparams.log ||
#    echo "$testname FAILED" >> ${testname}_ioparams.log
#mv $params{.bak,}

exit

