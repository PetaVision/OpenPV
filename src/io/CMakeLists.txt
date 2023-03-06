set (PVLibSrcCpp ${PVLibSrcCpp}
   ${SUBDIR}/ConfigParser.cpp
   ${SUBDIR}/Configuration.cpp
   ${SUBDIR}/fileio.cpp
   ${SUBDIR}/FileManager.cpp
   ${SUBDIR}/FileStream.cpp
   ${SUBDIR}/FileStreamBuilder.cpp
   ${SUBDIR}/io.cpp
   ${SUBDIR}/LayerBatchGatherScatter.cpp
   ${SUBDIR}/LayerFile.cpp
   ${SUBDIR}/LayerIO.cpp
   ${SUBDIR}/LocalPatchWeightsFile.cpp
   ${SUBDIR}/LocalPatchWeightsIO.cpp
   ${SUBDIR}/PVParams.cpp
   ${SUBDIR}/randomstateio.cpp
   ${SUBDIR}/SharedWeightsFile.cpp
   ${SUBDIR}/SharedWeightsIO.cpp
   ${SUBDIR}/SparseLayerBatchGatherScatter.cpp
   ${SUBDIR}/SparseLayerFile.cpp
   ${SUBDIR}/SparseLayerIO.cpp
   ${SUBDIR}/WeightsFileIO.cpp
)

set (PVLibSrcHpp ${PVLibSrcHpp}
   ${SUBDIR}/ConfigParser.hpp
   ${SUBDIR}/Configuration.hpp
   ${SUBDIR}/fileio.hpp
   ${SUBDIR}/FileManager.hpp
   ${SUBDIR}/FileStream.hpp
   ${SUBDIR}/FileStreamBuilder.hpp
   ${SUBDIR}/io.hpp
   ${SUBDIR}/LayerBatchGatherScatter.hpp
   ${SUBDIR}/LayerFile.hpp
   ${SUBDIR}/LayerIO.hpp
   ${SUBDIR}/LocalPatchWeightsFile.hpp
   ${SUBDIR}/LocalPatchWeightsIO.hpp
   ${SUBDIR}/PrintStream.hpp
   ${SUBDIR}/PVParams.hpp
   ${SUBDIR}/randomstateio.hpp
   ${SUBDIR}/SharedWeightsFile.hpp
   ${SUBDIR}/SharedWeightsIO.hpp
   ${SUBDIR}/SparseLayerBatchGatherScatter.hpp
   ${SUBDIR}/SparseLayerFile.hpp
   ${SUBDIR}/SparseLayerIO.hpp
   ${SUBDIR}/WeightsFile.hpp
   ${SUBDIR}/WeightsFileIO.hpp
)

if (PV_USE_MPI)
   set (PVLibSrcCpp ${PVLibSrcCpp} ${SUBDIR}/MPIRecvStream.cpp ${SUBDIR}/MPISendStream.cpp)
   set (PVLibSrcHpp ${PVLibSrcCpp} ${SUBDIR}/MPIRecvStream.hpp ${SUBDIR}/MPISendStream.hpp)
endif()
