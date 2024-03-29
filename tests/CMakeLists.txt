# This CMakeLists.txt file can be used to generate makefiles for several projects at once.
# It should be put in the directory containing the projects.

enable_testing()

# See http://www.cmake.org/Wiki/CMake_Useful_Variables for more variables to set
cmake_minimum_required(VERSION 3.0.0)

# Include the pv_add_test macro
include(PVAddTest)
include(PVAddExecutable)

set(TESTS_SHARED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Shared")

set(PV_MPI_SINGLE_PROCESS_TEST OFF CACHE BOOL "Defines whether single-process tests are run under MPI")
set(PV_SYSTEM_TEST_THREADS "" CACHE STRING "Defines the number of threads to use for system tests (empty string for max)")
set(PV_SYSTEM_TEST_COMMAND "" CACHE STRING "Defines the program that should invoke the test (if empty, ctest calls the test directly)")
set(PV_SYSTEM_TEST_SHUFFLE "" CACHE STRING "Defines whether to shuffle the order of param groups before running (empty string or \"0\" for no shuffling)")
set(PV_SYSTEM_TEST_MPIBLOCK ON CACHE BOOL "Defines whether to run system tests that use MPIBlocks with large numbers of processes")
set(PV_MPI_OPTIONS_EXTRA "" CACHE STRING "Options that ctest passes to the mpiexec program. Arguments are semicolon-separated.")

# Make sure that PV_SYSTEM_TEST_THREADS is a either a positive integer or the empty string
if("${PV_SYSTEM_TEST_THREADS}" MATCHES "^([1-9][0-9]*|)$")
   #Do nothing, correct
else("${PV_SYSTEM_TEST_THREADS}" MATCHES "^([1-9][0-9]*|)$")
   message(FATAL_ERROR "PV_SYSTEM_TEST_THREADS must be either a positive integer or the empty string")
endif("${PV_SYSTEM_TEST_THREADS}" MATCHES "^([1-9][0-9]*|)$")

# Make sure that PV_SYSTEM_TEST_SHUFFLE is either a nonnegative integer or the empty string
if("${PV_SYSTEM_TEST_SHUFFLE}" MATCHES "^(0|[1-9][0-9]*|)$")
   #Do nothing, correct
else("${PV_SYSTEM_TEST_SHUFFLE}" MATCHES "^(0|[1-9][0-9]*|)$")
   message(FATAL_ERROR "PV_SYSTEM_TEST_SHUFFLE must be either a positive integer or the empty string")
endif("${PV_SYSTEM_TEST_SHUFFLE}" MATCHES "^(0|[1-9][0-9]*|)$")

# Unit tests for individual classes happen first. If these fail, the rest of the results are unreliable.
add_subdirectory(ActivityBufferStatsProbeLocalTest)
add_subdirectory(BatchIndexerTest)
add_subdirectory(BufferTest)
add_subdirectory(BufferUtilsMPITest)
add_subdirectory(BufferUtilsPvpTest)
add_subdirectory(CheckpointEntryTest)
add_subdirectory(CheckpointerClassTest)
if (PV_USE_MPI)
   add_subdirectory(CheckpointerMPIBlockTest)
endif (PV_USE_MPI)
add_subdirectory(CheckpointWeightTest)
add_subdirectory(CheckStatsAllZeros)
add_subdirectory(ColumnEnergyOutputterTest)
add_subdirectory(ConfigParserTest)
add_subdirectory(DataStoreTest)
add_subdirectory(DeleteOlderCheckpointsTest)
add_subdirectory(EnergyProbeComponentTest)
add_subdirectory(ExpandLeadingTildeTest)
add_subdirectory(FileManagerTest)
add_subdirectory(ImageTest)
add_subdirectory(InputLayerNormalizeOffsetTest)
add_subdirectory(InputRegionLayerTest)
add_subdirectory(L1NormProbeLocalTest)
add_subdirectory(L2NormProbeAggregatorTest)
add_subdirectory(L2NormProbeLocalTest)
add_subdirectory(LayerFileTest)
add_subdirectory(LocalPatchWeightsFileTest)
add_subdirectory(MPIBlockTest)
if (PV_USE_MPI)
   add_subdirectory(MPISendStreamTest)
endif (PV_USE_MPI)
add_subdirectory(NormProbeAggregatorTest)
add_subdirectory(NormProbeOutputterTest)
add_subdirectory(PatchGeometryTest)
add_subdirectory(PathComponentsTest)
add_subdirectory(PostPatchSizeTest)
add_subdirectory(ProbeTriggerComponentTest)
add_subdirectory(ResponseTest)
add_subdirectory(SharedWeightsFileTest)
add_subdirectory(SparseLayerFileTest)
add_subdirectory(StatsProbeAggregatorTest)
add_subdirectory(StatsProbeLocalTest)
add_subdirectory(StatsProbeOutputterTest)
add_subdirectory(SynchronizeMarginsTest)
add_subdirectory(TargetLayerComponentTest)
add_subdirectory(TransposeWeightsTest)
add_subdirectory(WeightsClassTest)
add_subdirectory(WeightsFileIOTest)

# System tests that involve starting up a full instance of PetaVision happen next.
add_subdirectory(BasicSystemTest)
add_subdirectory(AdjustAxonalArborsTest)
add_subdirectory(ANNLayerVerticesTest)
add_subdirectory(ArborSystemTest)
add_subdirectory(AvgPoolTest)
add_subdirectory(BackgroundLayerTest)
add_subdirectory(BatchCheckpointSystemTest)
if (PV_USE_MPI)
   add_subdirectory(BatchMPICheckpointSystemTest)
endif (PV_USE_MPI)
add_subdirectory(BatchMethodTest)
add_subdirectory(BinningLayerTest)
add_subdirectory(ChannelCoefficientsTest)
add_subdirectory(CheckpointSystemTest)
add_subdirectory(CloneHyPerConnTest)
add_subdirectory(CloneKernelConnTest)
add_subdirectory(CloneVLayerTest)
add_subdirectory(CommandLineRestartTest)
add_subdirectory(ConfigFileSystemTest)
add_subdirectory(ConnectionRestartTest)
add_subdirectory(ConstantLayerTest)
add_subdirectory(ConvertToGrayscaleTest)
add_subdirectory(CopyConnTest)
add_subdirectory(DatastoreDelayTest)
add_subdirectory(DelaysToFeaturesTest)
add_subdirectory(DropoutLayerTest)
add_subdirectory(DryRunFlagTest)
add_subdirectory(FilenameParsingTest)
add_subdirectory(FirmThresholdCostTest)
add_subdirectory(GenericSystemTest)

if (PV_USE_CUDA)
   add_subdirectory(GPULCATest)
   add_subdirectory(GPUSystemTest)
endif (PV_USE_CUDA)

add_subdirectory(GroupNormalizationTest)
add_subdirectory(HyPerConnCheckpointerTest)
add_subdirectory(IdentConnTest)
add_subdirectory(ImageSystemTest)
add_subdirectory(ImageOffsetTest)
add_subdirectory(InputBCflagTest)
add_subdirectory(InputLayerNormalizeTest)
add_subdirectory(InputSystemTest)
add_subdirectory(ImportParamsTest)
add_subdirectory(InitializeFromCheckpointDirTest)
add_subdirectory(InitWeightsFileTest)
add_subdirectory(InitWeightsTest)
add_subdirectory(KernelTest)
add_subdirectory(KernelActivationTest)
add_subdirectory(L1NormProbeTest)
add_subdirectory(L2NormProbeTest)
add_subdirectory(LayerGeometryTest)
add_subdirectory(LayerPhaseTest)
add_subdirectory(LayerRestartTest)
add_subdirectory(LCATest)
add_subdirectory(LIFTest)
add_subdirectory(MarginWidthTest)
add_subdirectory(MaskLayerTest)
add_subdirectory(MaxPoolTest)
add_subdirectory(MomentumConnSimpleCheckpointerTest)
add_subdirectory(MomentumConnViscosityCheckpointerTest)
add_subdirectory(MomentumInitializeTest)
add_subdirectory(MomentumTest)
add_subdirectory(MomentumLCATest)
if (PV_USE_MPI)
   add_subdirectory(MPITest)
   add_subdirectory(MtoNOutputStateTest)
endif (PV_USE_MPI)
add_subdirectory(NegativeConnFrameNumberTest)
add_subdirectory(NegativeLayerFrameNumberTest)
add_subdirectory(NormalizeSubclassSystemTest)
add_subdirectory(NormalizeSystemTest)
add_subdirectory(ParameterSweepTest)
if (PV_USE_LUA)
  add_subdirectory(ParamsLuaTest)
endif (PV_USE_LUA)
add_subdirectory(PlasticCloneConnTest)
add_subdirectory(PlasticConnTest)
add_subdirectory(PlasticTransposeConnTest)
add_subdirectory(PoolingConnCheckpointerTest)
add_subdirectory(PoolingGPUTest)
add_subdirectory(ProbeOutputTest)
add_subdirectory(PtwiseQuotientLayerTest)
add_subdirectory(RandomOrderTest)
add_subdirectory(RandStateSystemTest)
add_subdirectory(ReceiveFromPostTest)
add_subdirectory(ReduceAcrossBatchTest)
add_subdirectory(RequireAllZeroActivityProbeTest)
add_subdirectory(RescaleLayerTest)
add_subdirectory(ResetStateOnTriggerTest)
add_subdirectory(SegmentTest)
add_subdirectory(ShrunkenPatchTest)
add_subdirectory(SparseIdentTest)
add_subdirectory(StatsProbeTest)
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
if (PV_USE_MPI)
   add_subdirectory(test_mpi_specifyrowscolumns)
endif (PV_USE_MPI)
add_subdirectory(test_nearby_neighbor)
add_subdirectory(test_patch_head)
add_subdirectory(test_sign)
add_subdirectory(TotalEnergyTest)
add_subdirectory(TransposeConnTest)
add_subdirectory(TransposeHyPerConnTest)
add_subdirectory(TriggerTest)
add_subdirectory(UnequalPatchSizeTest)
add_subdirectory(WriteActivitySparseTest)
add_subdirectory(WriteSparseFileTest)
add_subdirectory(WTAConnTest)
# # Insert additional add_subdirectory commands to add other projects.

#October 7, 2021: Many probe classes were deprecated.
# set(DEPRECATED_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../deprecated-tests)
# set(DEPRECATED_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/../deprecated-tests)
# add_subdirectory(${DEPRECATED_SOURCE_DIR}/PointProbeBatchTest ${DEPRECATED_BINARY_DIR}/PointProbeBatchTest)
# add_subdirectory(${DEPRECATED_SOURCE_DIR}/PointProbeTest ${DEPRECATED_BINARY_DIR}/PointProbeTest)
