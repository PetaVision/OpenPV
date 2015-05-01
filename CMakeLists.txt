# This CMakeLists.txt file can be used to generate makefiles for several projects at once.
# It should be put in the directory containing the projects.

project(PetaVision)

enable_testing()

# See http://www.cmake.org/Wiki/CMake_Useful_Variables for more variables to set
cmake_minimum_required(VERSION 2.8)

#Define function for adding a test to the test harness
macro(AddPVTest BaseName ParamNames inFlags inMpi inDepends)
   #Case based on size of paramNames
   list(LENGTH ${ParamNames} numParams)
   #No params case
   if(${numParams} EQUAL 0)
      set(testName ${BaseName})
      #One process
      add_test(${testName}_1 ${CMAKE_COMMAND} -E chdir ${BaseName}
         Debug/${BaseName} ${inFlags} -l
         ${testName}_1.log)
      if(${inMpi})
         #Two processes
         add_test(${testName}_2 ${CMAKE_COMMAND} -E chdir ${BaseName}
            mpirun -np 2 Debug/${BaseName} ${inFlags} -l
            ${testName}_2.log)
         #Four processes
         add_test(${testName}_4 ${CMAKE_COMMAND} -E chdir ${BaseName}
            mpirun -np 4 Debug/${BaseName} ${inFlags} -l
            ${testName}_4.log)
         #Add dependencies
         if(${inDepends})
            set_tests_properties(${testName}_2 PROPERTIES DEPENDS ${testName}_1)
            set_tests_properties(${testName}_4 PROPERTIES DEPENDS ${testName}_2)
         endif(${inDepends})
      endif(${inMpi})
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
         if(${inMpi})
            #Two processes
            add_test(${testName}_2 ${CMAKE_COMMAND} -E chdir ${BaseName}
               mpirun -np 2 Debug/${BaseName} -p input/${param}.params ${inFlags} -l
               ${testName}_2.log)
            #Four processes
            add_test(${testName}_4 ${CMAKE_COMMAND} -E chdir ${BaseName}
               mpirun -np 4 Debug/${BaseName} -p input/${param}.params ${inFlags} -l
               ${testName}_4.log)
            #Add dependencies
            if(${inDepends})
               set_tests_properties(${testName}_2 PROPERTIES DEPENDS ${testName}_1)
               set_tests_properties(${testName}_4 PROPERTIES DEPENDS ${testName}_2)
            endif(${inDepends})
         endif(${inMpi})
      endforeach(param)
   endif(${numParams} EQUAL 0)
endmacro(AddPVTest)

#Set CMAKE_MODULE_PATH
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PV_DIR}/docs/cmake/Modules/")

########
# March 31, 2015: renamed several cache variables to agree with C/C++ preprocessor directives
# so that they could be put into cMakeHeader.h easily instead of passing -D arguments to the compile command.
# Accordingly, test if old variable name exists, and if so, use its value as the new variable.
# then delete the old variable from the cache.  After a while, no one
# should have the old variables in the cache and this section can be deleted.
if (DEFINED OPEN_MP_THREADS)
   set(PV_USE_OPENMP_THREADS ${OPEN_MP_THREADS} CACHE BOOL "Defines whether PetaVision uses OpenMP threads")
   unset(OPEN_MP_THREADS CACHE)
endif (DEFINED OPEN_MP_THREADS)
if (DEFINED OPEN_CL_GPU)
   set(PV_USE_OPENCL ${OPEN_CL_GPU} CACHE BOOL "Defines whether PetaVision uses OpenCL GPU")
   unset(OPEN_CL_GPU CACHE)
endif (DEFINED OPEN_CL_GPU)
if (DEFINED CUDA_GPU)
   set(PV_USE_CUDA ${CUDA_GPU} CACHE BOOL "Defines whether PetaVision uses CUDA GPU")
   unset(CUDA_GPU CACHE)
endif (DEFINED CUDA_GPU)
if (DEFINED CUDNN)
   set(PV_USE_CUDNN ${CUDNN} CACHE BOOL "Defines whether PetaVision uses CUDA's cuDNN library")
   unset(CUDNN CACHE)
endif (DEFINED CUDNN)
########


# Set variable for current working directory (workspace directory)
set(PV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../PetaVision CACHE PATH "The base petavision directory")

set(PV_USE_OPENMP_THREADS FALSE CACHE BOOL "Defines if PetaVision uses OpenMP threads")
set(PV_USE_OPENCL FALSE CACHE BOOL "Defines if PetaVision uses OpenCL GPU")
set(PV_USE_CUDA FALSE CACHE BOOL "Defines if PetaVision uses CUDA GPU")
set(CUDA_RELEASE FALSE CACHE BOOL "Defines if Cuda compiles with optimization")
set(PV_USE_CUDNN FALSE CACHE BOOL "Defines if PetaVision uses Cuda's CUDNN library")
set(PV_SYSTEM_TEST_THREADS 1 CACHE STRING "Defines the number of threads to use for system tests")
set(CUDNN_PATH /nh/compneuro/Data/cuDNN/cudnn-6.5-linux-R1/ CACHE PATH "Location of cuDNN libraries")
set(OpenCL_dir /usr/local/cuda-6.5/ CACHE PATH "Location of opencl headers")


if(${PV_SYSTEM_TEST_THREADS} MATCHES "^([0-9]+)")
   #Do nothing, correct
else(${PV_SYSTEM_TEST_THREADS} MATCHES "^([0-9]+)")
   message(FATAL_ERROR "PV_SYSTEM_TEST_THREADS must be a positive integer")
endif(${PV_SYSTEM_TEST_THREADS} MATCHES "^([0-9]+)")

IF(PV_USE_CUDA AND PV_USE_OPENCL)
   message(FATAL_ERROR "PV_USE_CUDA and PV_USE_OPENCL cannot both be true")
ENDIF(PV_USE_CUDA AND PV_USE_OPENCL)

IF(NOT EXISTS "${PV_DIR}")
   message(FATAL_ERROR "PV_DIR \"${PV_DIR}\" does not exist.")
ENDIF(NOT EXISTS "${PV_DIR}")

if(PV_USE_OPENMP_THREADS AND NOT DEFINED PV_USE_OPENMP_THREADS_FLAGS_ADDED)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
   set(PV_USE_OPENMP_THREADS_FLAGS_ADDED TRUE)
endif (PV_USE_OPENMP_THREADS AND NOT DEFINED PV_USE_OPENMP_THREADS_FLAGS_ADDED)


add_subdirectory(AlexNetTest)
add_subdirectory(AdjustAxonalArborsTest)
add_subdirectory(ArborSystemTest)
add_subdirectory(BackgroundLayerTest)
add_subdirectory(BasicSystemTest)
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
add_subdirectory(MPITest2)
add_subdirectory(NormalizeSystemTest)
add_subdirectory(ParameterSweepTest)
add_subdirectory(PlasticConnTest)
add_subdirectory(PlasticTransposeConnTest)
add_subdirectory(PointProbeTest)
add_subdirectory(RandStateSystemTest)
add_subdirectory(ReadArborFileTest)
add_subdirectory(ReceiveFromPostTest)
add_subdirectory(RescaleLayerTest)
add_subdirectory(ShrunkenPatchTest)
add_subdirectory(ShrunkenPatchFlagTest)
add_subdirectory(SparsityProbeTest)
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
set(testParams BasicSystemTest) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName AdjustAxonalArborsTest) #The base name of the test
set(testParams AdjustAxonalArborsTest) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName ArborSystemTest) #The base name of the test
set(testParams test_arbors) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName BackgroundLayerTest) #The base name of the test
set(testParams repTest norepTest) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName BinningLayerTest) #The base name of the test
set(testParams BinningLayerTest) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName CheckpointSystemTest) #The base name of the test
set(testParams ) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends TRUE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName CloneHyPerConnTest) #The base name of the test
set(testParams CloneHyPerConnTest) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName CloneKernelConnTest) #The base name of the test
set(testParams CloneKernelConnTest) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName CloneVLayerTest) #The base name of the test
set(testParams ) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName ConnectionRestartTest) #The base name of the test
set(testParams ConnectionRestartTest) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName ConvertToGrayscaleTest) #The base name of the test
set(testParams ConvertToGrayscaleTest) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName CopyConnTest) #The base name of the test
set(testParams ) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName DatastoreDelayTest) #The base name of the test
set(testParams DatastoreDelayTest) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName DelaysToFeaturesTest) #The base name of the test
set(testParams test_delays) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})


set(testBaseName GenericSystemTest) #The base name of the test
set(testParams GenericSystemTest) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -c checkpoints/Checkpoint6 --testall -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends TRUE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName GPUSystemTest) #The base name of the test
set(testParams preTest postTest postTestNoTranspose postTestOneToMany
   postTest_linked postTestGpuGroup HyPerLCAGpuTest) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName GroupNormalizationTest) #The base name of the test
set(testParams GroupNormalizationTest) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName ImageOffsetTest) #The base name of the test
set(testParams ImageOffsetTest) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName ImageSystemTest) #The base name of the test
set(testParams multiframe_SystemTest) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName ImportParamsTest) #The base name of the test
set(testParams ImportParamsTest) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName ImprintConnTest) #The base name of the test
set(testParams ImprintConnTest) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName InitWeightsTest) #The base name of the test
set(testParams test_initweights) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName KernelActivationTest) #The base name of the test
set(testParams ) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName KernelTest) #The base name of the test
set(testParams test_kernel test_kernel_normalizepost_shrunken) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName LayerPhaseTest) #The base name of the test
set(testParams LayerPhaseTest) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName LayerRestartTest) #The base name of the test
set(testParams ) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends TRUE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName LCATest) #The base name of the test
set(testParams LCATest) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -c checkpoints/Checkpoint6 --testall -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends TRUE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName LIFTest) #The base name of the test
set(testParams LIFTest) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName MarginWidthTest) #The base name of the test
set(testParams MarginWidthTest) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName MaskLayerTest) #The base name of the test
set(testParams maskTest) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName MaxPoolTest) #The base name of the test
set(testParams maxpooltest gateTest) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName MovieSystemTest) #The base name of the test
set(testParams MovieSystemTest) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName MPITest2) #The base name of the test
set(testParams MPI_test) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName NormalizeSystemTest) #The base name of the test
set(testParams NormalizeSystemTest) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi FALSE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName ParameterSweepTest) #The base name of the test
set(testParams ParameterSweepTest) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName PlasticConnTest) #The base name of the test
set(testParams PlasticConnTest) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName PlasticTransposeConnTest) #The base name of the test
set(testParams PlasticTransposeConnTest) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName PointProbeTest) #The base name of the test
set(testParams PointProbeTest) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName RandStateSystemTest) #The base name of the test
set(testParams ) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends TRUE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName ReadArborFileTest) #The base name of the test
set(testParams ReadArborFileTest) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName ReceiveFromPostTest) #The base name of the test
set(testParams postTest_margins postTestNoTranspose 
   manyToOnePatchSizeTest oneToManyPatchSizeTest postTest_ManyToOne
   postTest_OneToMany) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName RescaleLayerTest) #The base name of the test
set(testParams RescaleLayerTest) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName ShrunkenPatchFlagTest) #The base name of the test
set(testParams ShrunkenPatchFlagTest) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName ShrunkenPatchTest) #The base name of the test
set(testParams ) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName StochasticReleaseTest) #The base name of the test
set(testParams StochasticReleaseTest) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName SumPoolTest) #The base name of the test
set(testParams sumpooltest unpoolTest) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName test_border_activity) #The base name of the test
set(testParams ) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName test_cocirc) #The base name of the test
set(testParams test_cocirc) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName test_constant_input) #The base name of the test
set(testParams test_constant_input) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName test_delta) #The base name of the test
set(testParams ) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi FALSE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName test_delta_pos) #The base name of the test
set(testParams ) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi FALSE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName test_extend_border) #The base name of the test
set(testParams ) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi FALSE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName test_kg) #The base name of the test
set(testParams ) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi FALSE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName test_kxpos) #The base name of the test
set(testParams ) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi FALSE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName test_kypos) #The base name of the test
set(testParams ) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi FALSE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName test_mirror_BCs) #The base name of the test
set(testParams test_mirror_BCs) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName test_mpi_specifyrowscolumns) #The base name of the test
set(testParams ) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
#This addtest is hard coded because this test requires a run with 6 mpi processes only
add_test(${testBaseName}_6 ${CMAKE_COMMAND} -E chdir ${testBaseName}
   mpirun -np 6 Debug/${testBaseName} ${testFlags} -l
   ${testBaseName}_6.log)

#Test from here
set(testBaseName test_nearby_neighbor) #The base name of the test
set(testParams ) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi FALSE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName test_patch_head) #The base name of the test
set(testParams ) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi FALSE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName test_post_weights) #The base name of the test
set(testParams test_post_weights) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName test_sign) #The base name of the test
set(testParams ) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi FALSE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName TriggerTest) #The base name of the test
set(testParams TriggerTest) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName TransposeConnTest) #The base name of the test
set(testParams ) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName TransposeHyPerConnTest) #The base name of the test
set(testParams ) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName UnequalPatchSizeTest) #The base name of the test
set(testParams UnequalPatchSizeTest) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName UpdateFromCloneTest) #The base name of the test
set(testParams updateFromCloneTest) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends FALSE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName WriteActivitySparseTest) #The base name of the test
set(testParams ) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends TRUE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

set(testBaseName WriteSparseFileTest) #The base name of the test
set(testParams WriteSparseFileTest) #Name of the parameter files in the input directory, leaving out .params
set(testFlags -c checkpoints/Checkpoint6 --testall -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to petavision test
set(testMpi TRUE) #Extra flags to pass to petavision test
set(testDepends TRUE) #This variable defines if the test is required to run sequentially
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi} ${testDepends})

