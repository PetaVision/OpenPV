set (PVLibSrcCpp ${PVLibSrcCpp}
   ${SUBDIR}/accumulate_functions.cpp
   ${SUBDIR}/BaseDelivery.cpp
   ${SUBDIR}/CloneDeliveryFacade.cpp
   ${SUBDIR}/HyPerDelivery.cpp
   ${SUBDIR}/HyPerDeliveryFacade.cpp
   ${SUBDIR}/IdentDelivery.cpp
   ${SUBDIR}/PoolingDelivery.cpp
   ${SUBDIR}/PostsynapticPerspectiveConvolveDelivery.cpp
#   ${SUBDIR}/PostsynapticPerspectiveGPUDelivery.cpp
   ${SUBDIR}/PostsynapticPerspectiveStochasticDelivery.cpp
   ${SUBDIR}/PresynapticPerspectiveConvolveDelivery.cpp
#   ${SUBDIR}/PresynapticPerspectiveGPUDelivery.cpp
   ${SUBDIR}/PresynapticPerspectiveStochasticDelivery.cpp
   ${SUBDIR}/RescaleDelivery.cpp
   ${SUBDIR}/TransposePoolingDelivery.cpp
)

set (PVLibSrcHpp ${PVLibSrcHpp}
   ${SUBDIR}/accumulate_functions.hpp
   ${SUBDIR}/BaseDelivery.hpp
   ${SUBDIR}/CloneDeliveryFacade.hpp
   ${SUBDIR}/HyPerDeliveryFacade.hpp
   ${SUBDIR}/HyPerDelivery.hpp
   ${SUBDIR}/IdentDelivery.hpp
   ${SUBDIR}/PoolingDelivery.hpp
   ${SUBDIR}/PostsynapticPerspectiveConvolveDelivery.hpp
#   ${SUBDIR}/PostsynapticPerspectiveGPUDelivery.hpp
   ${SUBDIR}/PostsynapticPerspectiveStochasticDelivery.hpp
   ${SUBDIR}/PresynapticPerspectiveConvolveDelivery.hpp
#   ${SUBDIR}/PresynapticPerspectiveGPUDelivery.hpp
   ${SUBDIR}/PresynapticPerspectiveStochasticDelivery.hpp
   ${SUBDIR}/RescaleDelivery.hpp
   ${SUBDIR}/TransposePoolingDelivery.hpp
)
