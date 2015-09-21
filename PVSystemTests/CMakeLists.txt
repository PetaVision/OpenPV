# This CMakeLists.txt file can be used to generate makefiles for several projects at once.
# It should be put in the directory containing the projects.

project(PVSystemTests)

enable_testing()

# See http://www.cmake.org/Wiki/CMake_Useful_Variables for more variables to set
cmake_minimum_required(VERSION 2.8.10)


# Set variable for current working directory (workspace directory)
set(PV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../PetaVision CACHE PATH "The base petavision directory")

IF(NOT EXISTS "${PV_DIR}")
   message(FATAL_ERROR "PV_DIR \"${PV_DIR}\" does not exist. Please set PV_DIR to the path containing the PetaVision lib and src
directories.")
ENDIF(NOT EXISTS "${PV_DIR}")

# Note: It would be preferable to parse ${PV_DIR}/src/include/cMakeHeader.h for these values.
set(PV_USE_MPI TRUE CACHE BOOL "Defines whether the MPI environment is present")
set(PV_USE_OPENMP_THREADS FALSE CACHE BOOL "Defines if PetaVision uses OpenMP threads")
set(PV_USE_OPENCL FALSE CACHE BOOL "Defines if PetaVision uses OpenCL GPU")
set(PV_USE_CUDA FALSE CACHE BOOL "Defines if PetaVision uses CUDA GPU")
set(CUDA_RELEASE FALSE CACHE BOOL "Defines if Cuda compiles with optimization")
set(PV_USE_CUDNN FALSE CACHE BOOL "Defines if PetaVision uses Cuda's CUDNN library")
set(PV_SYSTEM_TEST_THREADS 1 CACHE STRING "Defines the number of threads to use for system tests")
set(CUDNN_PATH /nh/compneuro/Data/cuDNN/cudnn-6.5-linux-R1/ CACHE PATH "Location of cuDNN libraries")
set(OpenCL_dir /usr/local/cuda-6.5/ CACHE PATH "Location of opencl headers")


#Define function for adding a test to the test harness
macro(AddPVTest BaseName ParamNames inFlags inMpi)
   #Case based on size of paramNames
   list(LENGTH ${ParamNames} numParams)
   #No params case
   if(${numParams} EQUAL 0)
      set(testName ${BaseName})
      #One process
      add_test(${testName}_1 ${CMAKE_COMMAND} -E chdir ${BaseName}
         Debug/${BaseName} ${inFlags} -l
         ${testName}_1.log)
      if(${PV_USE_MPI} AND ${inMpi})
         #Two processes
         add_test(${testName}_2 ${CMAKE_COMMAND} -E chdir ${BaseName}
            mpirun -np 2 Debug/${BaseName} ${inFlags} -l 
            ${testName}_2.log)
         #Four processes
         add_test(${testName}_4 ${CMAKE_COMMAND} -E chdir ${BaseName}
            mpirun -np 4 Debug/${BaseName} ${inFlags} -l 
            ${testName}_4.log)
         #Add dependencies
         set_tests_properties(${testName}_2 PROPERTIES DEPENDS ${testName}_1)
         set_tests_properties(${testName}_4 PROPERTIES DEPENDS ${testName}_2)
      endif(${PV_USE_MPI} AND ${inMpi})
   else(${numParams} EQUAL 0)
      #Multiple params here
      foreach(param IN LISTS ${ParamNames})
         #Set test name based on number of parameters
         if(${numParams} GREATER 1)
            set(testName ${BaseName}_${param})
         else(${numParams} GREATER 1)
            set(testName ${BaseName})
         endif(${numParams} GREATER 1)
         #One process
         add_test(${testName}_1 ${CMAKE_COMMAND} -E chdir ${BaseName}
            Debug/${BaseName} -p input/${param}.params ${inFlags} -l
            ${testName}_1.log)
         if(${PV_USE_MPI} AND ${inMpi})
            #Two processes
            add_test(${testName}_2 ${CMAKE_COMMAND} -E chdir ${BaseName}
               mpirun -np 2 Debug/${BaseName} -p input/${param}.params
               ${inFlags} -l
               ${testName}_2.log)
            #Four processes
            add_test(${testName}_4 ${CMAKE_COMMAND} -E chdir ${BaseName}
               mpirun -np 4 Debug/${BaseName} -p input/${param}.params
               ${inFlags} -l
               ${testName}_4.log)
            #Add dependencies
            set_tests_properties(${testName}_2 PROPERTIES DEPENDS ${testName}_1)
            set_tests_properties(${testName}_4 PROPERTIES DEPENDS ${testName}_2)
         endif(${PV_USE_MPI} AND ${inMpi})
      endforeach(param)
   endif(${numParams} EQUAL 0)
endmacro(AddPVTest)

#Set CMAKE_MODULE_PATH
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PV_DIR}/docs/cmake/Modules/")


if(${PV_SYSTEM_TEST_THREADS} MATCHES "^([0-9]+)")
   #Do nothing, correct
else(${PV_SYSTEM_TEST_THREADS} MATCHES "^([0-9]+)")
   message(FATAL_ERROR "PV_SYSTEM_TEST_THREADS must be a positive integer")
endif(${PV_SYSTEM_TEST_THREADS} MATCHES "^([0-9]+)")

IF(PV_USE_CUDA AND PV_USE_OPENCL)
   message(FATAL_ERROR "PV_USE_CUDA and PV_USE_OPENCL cannot both be true")
ENDIF(PV_USE_CUDA AND PV_USE_OPENCL)

if(PV_USE_OPENMP_THREADS AND NOT DEFINED PV_USE_OPENMP_THREADS_FLAGS_ADDED)
   if(APPLE AND CLANG_OMP)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp=libiomp5")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp=libiomp5")
      set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libstdc++")
   else(APPLE AND CLANG_OMP)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
   endif(APPLE AND CLANG_OMP)
   set(PV_USE_OPENMP_THREADS_FLAGS_ADDED TRUE)
endif (PV_USE_OPENMP_THREADS AND NOT DEFINED PV_USE_OPENMP_THREADS_FLAGS_ADDED)


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
#add_subdirectory(FourByFourGenerativeTest) # Moved to inactivesandboxes/SymmetryBreakingGenerative Apr 6, 2015
#add_subdirectory(FourByFourTopDownTest) # Moved to inactivesandboxes/SymmetryBreakingGenerative Apr 6, 2015
add_subdirectory(GenericSystemTest)
add_subdirectory(GPUSystemTest)
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
# add_subdirectory(MatchingPursuitTest) # Moved to auxlib/matchingpursuit Mar 18, 2015
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
       mpirun -np 2 Debug/${testBaseName} ${testFlags} -l
       ${testBaseName}_2.log)
    add_test(${testBaseName}_4 ${CMAKE_COMMAND} -E chdir ${testBaseName}
       mpirun -np 4 Debug/${testBaseName} ${testFlags} -l
       ${testBaseName}_4.log)
    add_test(${testBaseName}_8 ${CMAKE_COMMAND} -E chdir ${testBaseName}
       mpirun -np 8 Debug/${testBaseName} ${testFlags} -l
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
      mpirun -np 4 Debug/${testBaseName} ${testFlags} -p input/BatchSweepTest.params -l
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
      mpirun -np 10 Debug/${testBaseName} ${testFlags} -l
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

set(testBaseName GenericSystemTest) #The base name of the test
set(testParams GenericSystemTest) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -c checkpoints/Checkpoint6 --testall -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

set(testBaseName GPUSystemTest) #The base name of the test
set(testParams preTest postTest postTestNoTranspose postTestOneToMany
   postTest_linked postTestGpuGroup HyPerLCAGpuTest
   postRecvBatch preRecvBatch) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})

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
set(testParams LIFTest) #Names of the parameter files in the input directory, leaving out .params
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
set(testParams postTest_margins postTestNoTranspose 
   manyToOnePatchSizeTest oneToManyPatchSizeTest postTest_ManyToOne
   postTest_OneToMany) #Names of the parameter files in the input directory, leaving out .params
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
       mpirun -np 6 Debug/${testBaseName} ${testFlags} -l
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

