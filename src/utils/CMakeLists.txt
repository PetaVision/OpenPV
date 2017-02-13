set (PVLibSrcCpp ${PVLibSrcCpp}
   ${SUBDIR}/BorderExchange.cpp
   ${SUBDIR}/BufferUtilsRescale.cpp
   ${SUBDIR}/Clock.cpp
   ${SUBDIR}/PVAssert.cpp
   ${SUBDIR}/PVAlloc.cpp
   ${SUBDIR}/PVLog.cpp
   ${SUBDIR}/Timer.cpp
)

set (PVLibSrcHpp ${PVLibSrcHpp}
   ${SUBDIR}/BorderExchange.hpp
   ${SUBDIR}/BufferUtilsMPI.hpp
   ${SUBDIR}/BufferUtilsPvp.hpp
   ${SUBDIR}/BufferUtilsRescale.hpp
   ${SUBDIR}/Clock.hpp
   ${SUBDIR}/PVAssert.hpp
   ${SUBDIR}/PVAlloc.hpp
   ${SUBDIR}/PVLog.hpp
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
