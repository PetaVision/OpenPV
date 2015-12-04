# This CMakeLists.txt file can be used to generate makefiles for several projects at once.
# It should be put in the directory containing the projects.

enable_testing()

# See http://www.cmake.org/Wiki/CMake_Useful_Variables for more variables to set
cmake_minimum_required(VERSION 2.8.10)

# Include the AddPVTest macro
include(AddPVTest)

set(PV_SYSTEM_TEST_THREADS "" CACHE STRING "Defines the number of threads to use for system tests (empty string for max)")

# Make sure that PV_SYSTEM_TEST_THREADS is a number
if("${PV_SYSTEM_TEST_THREADS}" MATCHES "^([0-9]+|)$")
   #Do nothing, correct
else("${PV_SYSTEM_TEST_THREADS}" MATCHES "^([0-9]+|)$")
   message(FATAL_ERROR "PV_SYSTEM_TEST_THREADS must be either a positive integer or the empty string")
endif("${PV_SYSTEM_TEST_THREADS}" MATCHES "^([0-9]+|)$")

add_subdirectory(AdjustAxonalArborsTest)
add_subdirectory(ArborSystemTest)
add_subdirectory(BackgroundLayerTest)
add_subdirectory(BasicSystemTest)
add_subdirectory(BatchCheckpointSystemTest)
add_subdirectory(BatchMPICheckpointSystemTest)
add_subdirectory(BatchSweepTest)
add_subdirectory(BatchWeightUpdateTest)
add_subdirectory(BatchWeightUpdateMpiTest)
add_subdirectory(BinningLayerTest)
add_subdirectory(CheckpointSystemTest)
add_subdirectory(CloneHyPerConnTest)
add_subdirectory(CloneKernelConnTest)
add_subdirectory(CloneVLayerTest)
add_subdirectory(ConnectionRestartTest)
add_subdirectory(ConvertToGrayscaleTest)
add_subdirectory(CopyConnTest)
add_subdirectory(DatastoreDelayTest)
add_subdirectory(DelaysToFeaturesTest)
add_subdirectory(DtAdaptControllerTest)
add_subdirectory(GenericSystemTest)

if (PV_USE_CUDA OR PV_USE_OPENCL)
   add_subdirectory(GPUSystemTest)
endif (PV_USE_CUDA OR PV_USE_OPENCL)

add_subdirectory(GroupNormalizationTest)
add_subdirectory(ImageSystemTest)
add_subdirectory(ImageOffsetTest)
add_subdirectory(ImportParamsTest)
add_subdirectory(ImprintConnTest)
add_subdirectory(InitWeightsTest)
add_subdirectory(KernelTest)
add_subdirectory(KernelActivationTest)
add_subdirectory(LayerPhaseTest)
add_subdirectory(LayerRestartTest)
add_subdirectory(LCATest)
add_subdirectory(LIFTest)
add_subdirectory(MarginWidthTest)
add_subdirectory(MaskLayerTest)
add_subdirectory(MaxPoolTest)
add_subdirectory(MovieSystemTest)
add_subdirectory(MomentumTest)
#add_subdirectory(MLPTest) # Moved to auxlib/mlearning
add_subdirectory(MPITest)
add_subdirectory(NormalizeSubclassSystemTest)
add_subdirectory(NormalizeSystemTest)
add_subdirectory(ParameterSweepTest)
add_subdirectory(PlasticConnTest)
add_subdirectory(PlasticTransposeConnTest)
add_subdirectory(PointProbeTest)
add_subdirectory(RandStateSystemTest)
add_subdirectory(ReadArborFileTest)
add_subdirectory(ReceiveFromPostTest)
add_subdirectory(RescaleLayerTest)
add_subdirectory(ResetStateOnTriggerTest)
add_subdirectory(ShrunkenPatchTest)
add_subdirectory(ShrunkenPatchFlagTest)
add_subdirectory(StochasticReleaseTest)
add_subdirectory(SumPoolTest)
add_subdirectory(test_border_activity)
add_subdirectory(test_cocirc)
add_subdirectory(test_constant_input)
add_subdirectory(test_datatypes)
add_subdirectory(test_delta)
add_subdirectory(test_delta_pos)
add_subdirectory(test_extend_border)
add_subdirectory(test_gauss2d)
add_subdirectory(test_kg)
add_subdirectory(test_kxpos)
add_subdirectory(test_kypos)
add_subdirectory(test_mirror_BCs)
add_subdirectory(test_mpi_specifyrowscolumns)
add_subdirectory(test_nearby_neighbor)
add_subdirectory(test_patch_head)
add_subdirectory(test_post_weights)
add_subdirectory(test_read_header)
add_subdirectory(test_sign)
add_subdirectory(TotalEnergyTest)
add_subdirectory(TransposeConnTest)
add_subdirectory(TransposeHyPerConnTest)
add_subdirectory(TriggerTest)
add_subdirectory(UnequalPatchSizeTest)
add_subdirectory(UpdateFromCloneTest)
#add_subdirectory(WindowSystemTest)
add_subdirectory(WriteActivitySparseTest)
add_subdirectory(WriteSparseFileTest)
# # Insert additional add_subdirectory commands to add other projects.


#Add tests here

set(testBaseName BasicSystemTest) #The base name of the test
set(testParams BasicSystemTest) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName AdjustAxonalArborsTest) #The base name of the test
set(testParams AdjustAxonalArborsTest) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName ArborSystemTest) #The base name of the test
set(testParams test_arbors) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName BackgroundLayerTest) #The base name of the test
set(testParams repTest norepTest) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName BatchCheckpointSystemTest) #The base name of the test
set(testParams ) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName BatchMPICheckpointSystemTest) #The base name of the test
set(testParams ) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi FALSE) #Whether to call inside mpiexec with multiple processes
if (${PV_USE_MPI})
    #This addtest is hard coded because this test requires a run with 2, 4, and 8 mpi processes
    add_test(${testBaseName}_2 ${CMAKE_COMMAND} -E chdir ${testBaseName}
       ${MPIEXEC} -np 2 Debug/${testBaseName} ${testFlags} -l
       ${testBaseName}_2.log)
    add_test(${testBaseName}_4 ${CMAKE_COMMAND} -E chdir ${testBaseName}
       ${MPIEXEC} -np 4 Debug/${testBaseName} ${testFlags} -l
       ${testBaseName}_4.log)
    add_test(${testBaseName}_8 ${CMAKE_COMMAND} -E chdir ${testBaseName}
       ${MPIEXEC} -np 8 Debug/${testBaseName} ${testFlags} -l
       ${testBaseName}_8.log)
    #Add dependencies
    set_tests_properties(${testBaseName}_4 PROPERTIES DEPENDS ${testBaseName}_2)
    set_tests_properties(${testBaseName}_8 PROPERTIES DEPENDS ${testBaseName}_4)
endif (${PV_USE_MPI})

set(testBaseName BatchSweepTest) #The base name of the test
set(testParams BatchSweepTest) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS} -batchwidth 4) #Extra flags to pass to PetaVision test
set(testMpi FALSE) #Whether to call inside mpiexec with multiple processes
if (${PV_USE_MPI})
   #This addtest is hard coded because this test requires a run with exactly 4 mpi processes
   add_test(${testBaseName}_4 ${CMAKE_COMMAND} -E chdir ${testBaseName}
      ${MPIEXEC} -np 4 Debug/${testBaseName} ${testFlags} -p input/BatchSweepTest.params -l
      ${testBaseName}_4.log)
endif (${PV_USE_MPI})

set(testBaseName BatchWeightUpdateTest) #The base name of the test
set(testParams ) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

if (${PV_USE_MPI})
   set(testBaseName BatchWeightUpdateMpiTest) #The base name of the test
   set(testParams ) #Names of the parameter files in the input directory, leaving out .params
   set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
   set(testMpi FALSE) #Whether to call inside mpiexec with multiple processes
   #This addtest is hard coded because this test requires a run with exactly 10 mpi processes
   add_test(${testBaseName}_10 ${CMAKE_COMMAND} -E chdir ${testBaseName}
      ${MPIEXEC} -np 10 Debug/${testBaseName} ${testFlags} -l
      ${testBaseName}_10.log)
endif (${PV_USE_MPI})

set(testBaseName BinningLayerTest) #The base name of the test
set(testParams BinningLayerTest) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName CheckpointSystemTest) #The base name of the test
set(testParams ) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName CloneHyPerConnTest) #The base name of the test
set(testParams CloneHyPerConnTest) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName CloneKernelConnTest) #The base name of the test
set(testParams CloneKernelConnTest) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName CloneVLayerTest) #The base name of the test
set(testParams ) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName ConnectionRestartTest) #The base name of the test
set(testParams ConnectionRestartTest) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName ConvertToGrayscaleTest) #The base name of the test
set(testParams ConvertToGrayscaleTest) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName CopyConnTest) #The base name of the test
set(testParams ) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName DatastoreDelayTest) #The base name of the test
set(testParams DatastoreDelayTest) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName DelaysToFeaturesTest) #The base name of the test
set(testParams test_delays) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName DtAdaptControllerTest) #The base name of the test
set(testParams ) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName GenericSystemTest) #The base name of the test
set(testParams GenericSystemTest) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -c checkpoints/Checkpoint6 --testall -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

if (PV_USE_CUDA OR PV_USE_OPENCL)
   set(testBaseName GPUSystemTest) #The base name of the test
   set(testParams preTest postTest postTestNoTranspose postTestOneToMany
      postTest_linked postTestGpuGroup HyPerLCAGpuTest
      postRecvBatch preRecvBatch) #Names of the parameter files in the input directory, leaving out .params
   set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
   set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
   AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})
endif (PV_USE_CUDA OR PV_USE_OPENCL)

set(testBaseName GroupNormalizationTest) #The base name of the test
set(testParams GroupNormalizationTest) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName ImageOffsetTest) #The base name of the test
set(testParams ImageOffsetTest) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName ImageSystemTest) #The base name of the test
set(testParams ImageFileIO ImagePvpFileIO ImagePvpFileIOSparse MovieFileIO MoviePvpFileIO) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName ImportParamsTest) #The base name of the test
set(testParams ImportParamsTest) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName ImprintConnTest) #The base name of the test
set(testParams ImprintConnTest) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName InitWeightsTest) #The base name of the test
set(testParams test_initweights) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName KernelActivationTest) #The base name of the test
set(testParams ) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName KernelTest) #The base name of the test
set(testParams test_kernel test_kernel_normalizepost test_kernel_normalizepost_shrunken) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName LayerPhaseTest) #The base name of the test
set(testParams LayerPhaseTest) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName LayerRestartTest) #The base name of the test
set(testParams ) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName LCATest) #The base name of the test
set(testParams LCATest) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -c checkpoints/Checkpoint6 --testall -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName LIFTest) #The base name of the test
if (PV_USE_CUDA OR PV_USE_OPENCL)
   set(testParams LIFTest LIFTest_GPU) #Names of the parameter files in the input directory, leaving out .params
else (PV_USE_CUDA OR PV_USE_OPENCL)
   set(testParams LIFTest) #Names of the parameter files in the input directory, leaving out .params
endif (PV_USE_CUDA OR PV_USE_OPENCL)
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName MarginWidthTest) #The base name of the test
set(testParams MarginWidthTest) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName MaskLayerTest) #The base name of the test
set(testParams maskTest) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName MaxPoolTest) #The base name of the test
set(testParams maxpooltest gateTest) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName MovieSystemTest) #The base name of the test
set(testParams MovieSystemTest) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName MPITest) #The base name of the test
set(testParams MPI_test) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName NormalizeSubclassSystemTest) #The base name of the test
set(testParams NormalizeSubclassSystemTest) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi FALSE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName NormalizeSystemTest) #The base name of the test
set(testParams NormalizeSystemTest) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi FALSE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName ParameterSweepTest) #The base name of the test
set(testParams ParameterSweepTest) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName PlasticConnTest) #The base name of the test
set(testParams PlasticConnTest) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName PlasticTransposeConnTest) #The base name of the test
set(testParams PlasticTransposeConnTest) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName PointProbeTest) #The base name of the test
set(testParams PointProbeTest) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName RandStateSystemTest) #The base name of the test
set(testParams ) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName ReadArborFileTest) #The base name of the test
set(testParams ReadArborFileTest) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName ReceiveFromPostTest) #The base name of the test
if (PV_USE_CUDA OR PV_USE_OPENCL)
   set(testParams postTest_margins postTestNoTranspose postTestNoTranspose_GPU
      manyToOnePatchSizeTest oneToManyPatchSizeTest postTest_ManyToOne
      postTest_OneToMany) #Names of the parameter files in the input directory, leaving out .params
else (PV_USE_CUDA OR PV_USE_OPENCL)
   set(testParams postTest_margins postTestNoTranspose 
      manyToOnePatchSizeTest oneToManyPatchSizeTest postTest_ManyToOne
      postTest_OneToMany) #Names of the parameter files in the input directory, leaving out .params
endif (PV_USE_CUDA OR PV_USE_OPENCL)
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName RescaleLayerTest) #The base name of the test
set(testParams RescaleLayerTest) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName ResetStateOnTriggerTest) #The base name of the test
set(testParams ResetStateOnTriggerTest) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName ShrunkenPatchFlagTest) #The base name of the test
set(testParams ShrunkenPatchFlagTest) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName ShrunkenPatchTest) #The base name of the test
set(testParams ) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName StochasticReleaseTest) #The base name of the test
set(testParams StochasticReleaseTest) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName SumPoolTest) #The base name of the test
set(testParams sumpooltest unpoolTest) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName test_border_activity) #The base name of the test
set(testParams ) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName test_cocirc) #The base name of the test
set(testParams test_cocirc) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName test_constant_input) #The base name of the test
set(testParams test_constant_input) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName test_delta) #The base name of the test
set(testParams ) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi FALSE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName test_delta_pos) #The base name of the test
set(testParams ) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi FALSE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName test_extend_border) #The base name of the test
set(testParams ) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi FALSE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName test_kg) #The base name of the test
set(testParams ) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi FALSE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName test_kxpos) #The base name of the test
set(testParams ) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi FALSE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName test_kypos) #The base name of the test
set(testParams ) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi FALSE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName test_mirror_BCs) #The base name of the test
set(testParams test_mirror_BCs) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName test_mpi_specifyrowscolumns) #The base name of the test
set(testParams ) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
if (${PV_USE_MPI})
    #This addtest is hard coded because this test requires a run with exactly 6 mpi processes
    add_test(${testBaseName}_6 ${CMAKE_COMMAND} -E chdir ${testBaseName}
       ${MPIEXEC} -np 6 Debug/${testBaseName} ${testFlags} -l
       ${testBaseName}_6.log)
endif (${PV_USE_MPI})

#Test from here
set(testBaseName test_nearby_neighbor) #The base name of the test
set(testParams ) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi FALSE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName test_patch_head) #The base name of the test
set(testParams ) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi FALSE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName test_post_weights) #The base name of the test
set(testParams test_post_weights) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName test_sign) #The base name of the test
set(testParams ) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi FALSE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName TriggerTest) #The base name of the test
set(testParams TriggerTest) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName TotalEnergyTest) #The base name of the test
set(testParams TotalEnergyTest) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -c checkpoints/Checkpoint6 --testall -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName TransposeConnTest) #The base name of the test
set(testParams ) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName TransposeHyPerConnTest) #The base name of the test
set(testParams ) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName UnequalPatchSizeTest) #The base name of the test
set(testParams UnequalPatchSizeTest) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName UpdateFromCloneTest) #The base name of the test
set(testParams updateFromCloneTest) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName WriteActivitySparseTest) #The base name of the test
set(testParams ) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName WriteSparseFileTest) #The base name of the test
set(testParams WriteSparseFileTest) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -c checkpoints/Checkpoint6 --testall -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

