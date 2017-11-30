set (PVLibSrcCpp ${PVLibSrcCpp}
   ${SUBDIR}/BaseDelivery.cpp
   ${SUBDIR}/HyPerDelivery.cpp
   ${SUBDIR}/HyPerDeliveryFacade.cpp
   ${SUBDIR}/IdentDelivery.cpp
   ${SUBDIR}/PostsynapticPerspectiveConvolveDelivery.cpp
#   ${SUBDIR}/PostsynapticPerspectiveGPUDelivery.cpp
   ${SUBDIR}/PostsynapticPerspectiveStochasticDelivery.cpp
   ${SUBDIR}/PresynapticPerspectiveConvolveDelivery.cpp
#   ${SUBDIR}/PresynapticPerspectiveGPUDelivery.cpp
   ${SUBDIR}/PresynapticPerspectiveStochasticDelivery.cpp
   ${SUBDIR}/RescaleDelivery.cpp
)

set (PVLibSrcHpp ${PVLibSrcHpp}
   ${SUBDIR}/BaseDelivery.hpp
   ${SUBDIR}/HyPerDeliveryFacade.hpp
   ${SUBDIR}/HyPerDelivery.hpp
   ${SUBDIR}/IdentDelivery.hpp
   ${SUBDIR}/PostsynapticPerspectiveConvolveDelivery.hpp
#   ${SUBDIR}/PostsynapticPerspectiveGPUDelivery.hpp
   ${SUBDIR}/PostsynapticPerspectiveStochasticDelivery.hpp
   ${SUBDIR}/PresynapticPerspectiveConvolveDelivery.hpp
#   ${SUBDIR}/PresynapticPerspectiveGPUDelivery.hpp
   ${SUBDIR}/PresynapticPerspectiveStochasticDelivery.hpp
   ${SUBDIR}/RescaleDelivery.hpp
)
