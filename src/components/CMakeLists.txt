set (PVLibSrcCpp ${PVLibSrcCpp}
   ${SUBDIR}/LogTimeScaleController.cpp
   ${SUBDIR}/KneeTimeScaleController.cpp
   ${SUBDIR}/AdaptiveTimeScaleController.cpp
   ${SUBDIR}/BatchIndexer.cpp
   ${SUBDIR}/CloneWeightsPair.cpp
   ${SUBDIR}/ConnectionData.cpp
   ${SUBDIR}/ImpliedWeights.cpp
   ${SUBDIR}/NoCheckpointConnectionData.cpp
   ${SUBDIR}/PatchGeometry.cpp
   ${SUBDIR}/PostWeights.cpp
   ${SUBDIR}/Weights.cpp
   ${SUBDIR}/WeightsPair.cpp
)

set (PVLibSrcHpp ${PVLibSrcHpp}
   ${SUBDIR}/LogTimeScaleController.cpp
   ${SUBDIR}/KneeTimeScaleController.cpp
   ${SUBDIR}/AdaptiveTimeScaleController.hpp
   ${SUBDIR}/BatchIndexer.hpp
   ${SUBDIR}/CloneWeightsPair.hpp
   ${SUBDIR}/ConnectionData.hpp
   ${SUBDIR}/ImpliedWeights.hpp
   ${SUBDIR}/NoCheckpointConnectionData.hpp
   ${SUBDIR}/Patch.hpp
   ${SUBDIR}/PatchGeometry.hpp
   ${SUBDIR}/PostWeights.hpp
   ${SUBDIR}/Weights.hpp
   ${SUBDIR}/WeightsPair.hpp
)
