set (PVLibSrcCpp ${PVLibSrcCpp}
   ${SUBDIR}/BroadcastLayerFile.cpp
   ${SUBDIR}/BroadcastPreWeightsFile.cpp
   ${SUBDIR}/BroadcastPreWeightsIO.cpp
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
   ${SUBDIR}/PVPFrameIndexer.cpp
   ${SUBDIR}/randomstateio.cpp
   ${SUBDIR}/SharedWeightsFile.cpp
   ${SUBDIR}/SharedWeightsIO.cpp
   ${SUBDIR}/SparseBroadcastLayerFile.cpp
   ${SUBDIR}/SparseLayerBatchGatherScatter.cpp
   ${SUBDIR}/SparseLayerFile.cpp
   ${SUBDIR}/SparseLayerIO.cpp
)

set (PVLibSrcHpp ${PVLibSrcHpp}
   ${SUBDIR}/BroadcastLayerFile.hpp
   ${SUBDIR}/BroadcastPreWeightsFile.hpp
   ${SUBDIR}/BroadcastPreWeightsIO.hpp
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
   ${SUBDIR}/PVPFrameIndexer.hpp
   ${SUBDIR}/randomstateio.hpp
   ${SUBDIR}/SharedWeightsFile.hpp
   ${SUBDIR}/SharedWeightsIO.hpp
   ${SUBDIR}/SparseBroadcastLayerFile.hpp
   ${SUBDIR}/SparseLayerBatchGatherScatter.hpp
   ${SUBDIR}/SparseLayerFile.hpp
   ${SUBDIR}/SparseLayerIO.hpp
   ${SUBDIR}/WeightsFile.hpp
)

if (PV_USE_MPI)
   set (PVLibSrcCpp ${PVLibSrcCpp} ${SUBDIR}/MPIRecvStream.cpp ${SUBDIR}/MPISendStream.cpp)
   set (PVLibSrcHpp ${PVLibSrcCpp} ${SUBDIR}/MPIRecvStream.hpp ${SUBDIR}/MPISendStream.hpp)
endif()
