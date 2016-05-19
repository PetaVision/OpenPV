set (PVLibSrcCpp ${PVLibSrcCpp}
   ${SUBDIR}/InputComponent.cpp
   ${SUBDIR}/StateComponent.cpp
   ${SUBDIR}/LCAStateComponent.cpp
   ${SUBDIR}/OutputComponent.cpp
   ${SUBDIR}/FileOutputComponent.cpp
   ${SUBDIR}/ImageOutputComponent.cpp
   ${SUBDIR}/AudioOutputComponent.cpp
   ${SUBDIR}/SigmoidOutputComponent.cpp
   ${SUBDIR}/TanhOutputComponent.cpp
   ${SUBDIR}/ANNOutputComponent.cpp
)

set (PVLibSrcHpp ${PVLibSrcHpp}
   ${SUBDIR}/Component.hpp
   ${SUBDIR}/InputComponent.hpp
   ${SUBDIR}/StateComponent.hpp
   ${SUBDIR}/LCAStateComponent.hpp
   ${SUBDIR}/OutputComponent.hpp
   ${SUBDIR}/FileOutputComponent.hpp
   ${SUBDIR}/ImageOutputComponent.hpp
   ${SUBDIR}/AudioOutputComponent.hpp
   ${SUBDIR}/SigmoidOutputComponent.hpp
   ${SUBDIR}/TanhOutputComponent.hpp
   ${SUBDIR}/ANNOutputComponent.hpp
)
