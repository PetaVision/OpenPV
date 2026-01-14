set (PVLibSrcCpp ${PVLibSrcCpp}
   ${SUBDIR}/BatchIndexer.cpp
   ${SUBDIR}/Image.cpp
   ${SUBDIR}/ImpliedWeights.cpp
   ${SUBDIR}/MPIBlock.cpp
   ${SUBDIR}/PatchGeometry.cpp
   ${SUBDIR}/WeightData.cpp
   ${SUBDIR}/Weights.cpp
)

set (PVLibSrcHpp ${PVLibSrcHpp}
   ${SUBDIR}/BatchIndexer.hpp
   ${SUBDIR}/Buffer.hpp
   ${SUBDIR}/Image.hpp
   ${SUBDIR}/ImpliedWeights.hpp
   ${SUBDIR}/MPIBlock.hpp
   ${SUBDIR}/Patch.hpp
   ${SUBDIR}/PatchGeometry.hpp
   ${SUBDIR}/PVLayerLoc.hpp
   ${SUBDIR}/RingBuffer.hpp
   ${SUBDIR}/WeightData.hpp
   ${SUBDIR}/Weights.hpp
)

set (PVLibSrcHpp ${PVLibSrcTpp}
   ${SUBDIR}/Buffer.tpp
)
