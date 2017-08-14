set (PVLibSrcCpp ${PVLibSrcCpp}
   ${SUBDIR}/LogTimeScaleController.cpp
   ${SUBDIR}/KneeTimeScaleController.cpp
   ${SUBDIR}/AdaptiveTimeScaleController.cpp
   ${SUBDIR}/BatchIndexer.cpp
   ${SUBDIR}/PatchGeometry.cpp
)

set (PVLibSrcHpp ${PVLibSrcHpp}
   ${SUBDIR}/LogTimeScaleController.cpp
   ${SUBDIR}/KneeTimeScaleController.cpp
   ${SUBDIR}/AdaptiveTimeScaleController.hpp
   ${SUBDIR}/BatchIndexer.hpp
   ${SUBDIR}/Patch.hpp
   ${SUBDIR}/PatchGeometry.hpp
)
