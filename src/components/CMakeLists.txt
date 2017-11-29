set (PVLibSrcCpp ${PVLibSrcCpp}
   ${SUBDIR}/BaseDelivery.cpp
   ${SUBDIR}/LogTimeScaleController.cpp
   ${SUBDIR}/KneeTimeScaleController.cpp
   ${SUBDIR}/AdaptiveTimeScaleController.cpp
   ${SUBDIR}/BatchIndexer.cpp
   ${SUBDIR}/ConnectionData.cpp
   ${SUBDIR}/HyPerDelivery.cpp
   ${SUBDIR}/HyPerDeliveryFacade.cpp
   ${SUBDIR}/IdentDelivery.cpp
   ${SUBDIR}/ImpliedWeights.cpp
   ${SUBDIR}/PatchGeometry.cpp
   ${SUBDIR}/PostsynapticPerspectiveConvolveDelivery.cpp
#   ${SUBDIR}/PostsynapticPerspectiveGPUDelivery.cpp
   ${SUBDIR}/PostsynapticPerspectiveStochasticDelivery.cpp
   ${SUBDIR}/PostWeights.cpp
   ${SUBDIR}/PresynapticPerspectiveConvolveDelivery.cpp
#   ${SUBDIR}/PresynapticPerspectiveGPUDelivery.cpp
   ${SUBDIR}/PresynapticPerspectiveStochasticDelivery.cpp
   ${SUBDIR}/RescaleDelivery.cpp
   ${SUBDIR}/Weights.cpp
   ${SUBDIR}/WeightsPair.cpp
)

set (PVLibSrcHpp ${PVLibSrcHpp}
   ${SUBDIR}/BaseDelivery.hpp
   ${SUBDIR}/LogTimeScaleController.cpp
   ${SUBDIR}/KneeTimeScaleController.cpp
   ${SUBDIR}/AdaptiveTimeScaleController.hpp
   ${SUBDIR}/BatchIndexer.hpp
   ${SUBDIR}/ConnectionData.hpp
   ${SUBDIR}/HyPerDelivery.hpp
   ${SUBDIR}/HyPerDeliveryFacade.hpp
   ${SUBDIR}/IdentDelivery.hpp
   ${SUBDIR}/ImpliedWeights.hpp
   ${SUBDIR}/Patch.hpp
   ${SUBDIR}/PatchGeometry.hpp
   ${SUBDIR}/PostsynapticPerspectiveConvolveDelivery.hpp
#   ${SUBDIR}/PostsynapticPerspectiveGPUDelivery.hpp
   ${SUBDIR}/PostsynapticPerspectiveStochasticDelivery.hpp
   ${SUBDIR}/PostWeights.hpp
   ${SUBDIR}/PresynapticPerspectiveConvolveDelivery.hpp
#   ${SUBDIR}/PresynapticPerspectiveGPUDelivery.hpp
   ${SUBDIR}/PresynapticPerspectiveStochasticDelivery.hpp
   ${SUBDIR}/RescaleDelivery.hpp
   ${SUBDIR}/Weights.hpp
   ${SUBDIR}/WeightsPair.hpp
)
