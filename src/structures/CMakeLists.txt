set (PVLibSrcCpp ${PVLibSrcCpp}
   ${SUBDIR}/Image.cpp
   ${SUBDIR}/MPIBlock.cpp
   ${SUBDIR}/PatchGeometry.cpp
   ${SUBDIR}/WeightData.cpp
)

set (PVLibSrcHpp ${PVLibSrcHpp}
   ${SUBDIR}/Buffer.hpp
   ${SUBDIR}/Image.hpp
   ${SUBDIR}/MPIBlock.hpp
   ${SUBDIR}/Patch.hpp
   ${SUBDIR}/PatchGeometry.hpp
   ${SUBDIR}/RingBuffer.hpp
   ${SUBDIR}/WeightData.hpp
)

set (PVLibSrcHpp ${PVLibSrcTpp}
   ${SUBDIR}/Buffer.tpp
)
