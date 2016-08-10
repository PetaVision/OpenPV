set (PVLibSrcCpp ${PVLibSrcCpp}
   ${SUBDIR}/Timer.cpp
   ${SUBDIR}/PVAssert.cpp
   ${SUBDIR}/PVAlloc.cpp
   ${SUBDIR}/PVLog.cpp
   ${SUBDIR}/Image.cpp
   ${SUBDIR}/Buffer.cpp
)

set (PVLibSrcHpp ${PVLibSrcHpp}
   ${SUBDIR}/Timer.hpp
   ${SUBDIR}/PVAssert.hpp
   ${SUBDIR}/PVAlloc.hpp
   ${SUBDIR}/PVLog.hpp
   ${SUBDIR}/Image.hpp
   ${SUBDIR}/Buffer.hpp
)

set (PVLibSrcC ${PVLibSrcC}
   ${SUBDIR}/cl_random.c
   ${SUBDIR}/conversions.c
)

set (PVLibSrcH ${PVLibSrcH}
   ${SUBDIR}/cl_random.h
   ${SUBDIR}/conversions.h
)

