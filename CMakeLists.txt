# This CMakeLists.txt file can be used to generate makefiles for several projects at once.
# It should be put in the directory containing the projects.

project(PetaVision)

# See http://www.cmake.org/Wiki/CMake_Useful_Variables for more variables to set
cmake_minimum_required(VERSION 2.6)

# Set variable for current working directory (workspace directory)
if(NOT PV_DIR)
   set(PV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../PetaVision CACHE FILEPATH "The base petavision directory")
endif (NOT PV_DIR)
   


# The user may specify a non-standard compiler name for MPI
#  For example, the user may type:
#    cmake CMakeLists.txt -DCMAKE_C_COMPILER=openmpicc -DCMAKE_CXX_COMPILER=openmpic++
#    

add_subdirectory(AdjustAxonalArborsTest)
add_subdirectory(ArborSystemTest)
add_subdirectory(BasicSystemTest)
add_subdirectory(BinningLayerTest)
add_subdirectory(CheckpointSystemTest)
add_subdirectory(CloneKernelConnTest)
add_subdirectory(CloneVLayerTest)
add_subdirectory(DatastoreDelayTest)
add_subdirectory(DelaysToFeaturesTest)
add_subdirectory(FourByFourGenerativeTest)
add_subdirectory(FourByFourTopDownTest)
add_subdirectory(GenerativeConnTest)
add_subdirectory(GPUSystemTest)
add_subdirectory(ImageSystemTest)
add_subdirectory(ImportParamsTest)
add_subdirectory(InitWeightsTest)
add_subdirectory(KernelTest)
add_subdirectory(LayerPhaseTest)
add_subdirectory(LayerRestartTest)
add_subdirectory(LIFTest)
add_subdirectory(MarginWidthTest)
add_subdirectory(MatchingPursuitTest)
add_subdirectory(MovieSystemTest)
add_subdirectory(MPITest2)
add_subdirectory(NormalizeSystemTest)
add_subdirectory(ParameterSweepTest)
add_subdirectory(PlasticConnTest)
add_subdirectory(RandStateSystemTest)
add_subdirectory(ReadArborFileTest)
add_subdirectory(ReceiveFromPostTest)
add_subdirectory(ShrunkenPatchTest)
add_subdirectory(ShrunkenPatchFlagTest)
add_subdirectory(StochasticReleaseTest)
add_subdirectory(TransposeConnTest)
add_subdirectory(TriggerTest)
add_subdirectory(WindowSystemTest)
# # Insert additional add_subdirectory commands to add other projects.
#add_subdirectory(PVSystemTests/BasicSystemTest)
