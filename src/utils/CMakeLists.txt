set (PVLibSrcCpp ${PVLibSrcCpp}
   ${SUBDIR}/BorderExchange.cpp
   ${SUBDIR}/BufferUtilsPvp.cpp
   ${SUBDIR}/BufferUtilsRescale.cpp
   ${SUBDIR}/Clock.cpp
   ${SUBDIR}/CompareParamsFiles.cpp
   ${SUBDIR}/conversions.hpp
   ${SUBDIR}/ExpandLeadingTilde.cpp
   ${SUBDIR}/PVAssert.cpp
   ${SUBDIR}/PVAlloc.cpp
   ${SUBDIR}/PVLog.cpp
   ${SUBDIR}/Timer.cpp
   ${SUBDIR}/TransposeWeights.cpp
)

set (PVLibSrcHpp ${PVLibSrcHpp}
   ${SUBDIR}/BorderExchange.hpp
   ${SUBDIR}/BufferUtilsMPI.hpp
   ${SUBDIR}/BufferUtilsPvp.hpp
   ${SUBDIR}/BufferUtilsRescale.hpp
   ${SUBDIR}/Clock.hpp
   ${SUBDIR}/CompareParamsFiles.hpp
   ${SUBDIR}/ExpandLeadingTilde.hpp
   ${SUBDIR}/PVAssert.hpp
   ${SUBDIR}/PVAlloc.hpp
   ${SUBDIR}/PVLog.hpp
   ${SUBDIR}/Timer.hpp
   ${SUBDIR}/TransposeWeights.hpp
)

set (PVLibSrcHpp ${PVLibSrcTpp}
   ${SUBDIR}/BufferUtilsMPI.tpp
   ${SUBDIR}/BufferUtilsPvp.tpp
)

set (PVLibSrcC ${PVLibSrcC}
   ${SUBDIR}/cl_random.c
)

set (PVLibSrcH ${PVLibSrcH}
   ${SUBDIR}/cl_random.h
)
