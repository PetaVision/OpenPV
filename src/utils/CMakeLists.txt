set (PVLibSrcCpp ${PVLibSrcCpp}
   ${SUBDIR}/PVAssert.cpp
   ${SUBDIR}/PVAlloc.cpp
   ${SUBDIR}/PVLog.cpp
   ${SUBDIR}/Image.cpp
   ${SUBDIR}/Buffer.cpp
   ${SUBDIR}/BatchIndexer.cpp
   ${SUBDIR}/Timer.cpp
)

set (PVLibSrcHpp ${PVLibSrcHpp}
   ${SUBDIR}/PVAssert.hpp
   ${SUBDIR}/PVAlloc.hpp
   ${SUBDIR}/PVLog.hpp
   ${SUBDIR}/Image.hpp
   ${SUBDIR}/Buffer.hpp
   ${SUBDIR}/BatchIndexer.cpp
   ${SUBDIR}/RingBuffer.hpp
   ${SUBDIR}/Timer.hpp
)

set (PVLibSrcC ${PVLibSrcC}
   ${SUBDIR}/cl_random.c
   ${SUBDIR}/conversions.c
)

set (PVLibSrcH ${PVLibSrcH}
   ${SUBDIR}/cl_random.h
   ${SUBDIR}/conversions.h
)
