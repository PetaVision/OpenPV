set (PVLibSrcCpp ${PVLibSrcCpp}
   ${SUBDIR}/LogTimeScaleController.cpp
   ${SUBDIR}/KneeTimeScaleController.cpp
   ${SUBDIR}/AdaptiveTimeScaleController.cpp
   ${SUBDIR}/BatchIndexer.cpp
   ${SUBDIR}/ImpliedWeights.cpp
   ${SUBDIR}/PatchGeometry.cpp
   ${SUBDIR}/Weights.cpp
)

set (PVLibSrcHpp ${PVLibSrcHpp}
   ${SUBDIR}/LogTimeScaleController.cpp
   ${SUBDIR}/KneeTimeScaleController.cpp
   ${SUBDIR}/AdaptiveTimeScaleController.hpp
   ${SUBDIR}/BatchIndexer.hpp
   ${SUBDIR}/ImpliedWeights.hpp
   ${SUBDIR}/Patch.hpp
   ${SUBDIR}/PatchGeometry.hpp
   ${SUBDIR}/Weights.hpp
)
