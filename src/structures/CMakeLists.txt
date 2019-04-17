set (PVLibSrcCpp ${PVLibSrcCpp}
   ${SUBDIR}/Image.cpp
   ${SUBDIR}/MPIBlock.cpp
)

set (PVLibSrcHpp ${PVLibSrcHpp}
   ${SUBDIR}/Image.hpp
   ${SUBDIR}/MPIBlock.hpp
   ${SUBDIR}/Buffer.hpp
   ${SUBDIR}/RingBuffer.hpp
)

set (PVLibSrcHpp ${PVLibSrcTpp}
   ${SUBDIR}/Buffer.tpp
)
