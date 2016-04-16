set (PVLibSrcCpp ${PVLibSrcCpp}
   ${SUBDIR}/BasePVObject.cpp
   ${SUBDIR}/buildandrun.cpp
   ${SUBDIR}/Communicator.cpp
   ${SUBDIR}/DataStore.cpp
   ${SUBDIR}/GaussianRandom.cpp
   ${SUBDIR}/HyPerCol.cpp
   ${SUBDIR}/InterColComm.cpp
   ${SUBDIR}/PV_Arguments.cpp
   ${SUBDIR}/PV_Factory.cpp
   ${SUBDIR}/PV_Init.cpp
   ${SUBDIR}/PV_KeywordHandler.cpp
   ${SUBDIR}/Random.cpp
)

set (PVLibSrcHpp ${PVLibSrcHpp}
   ${SUBDIR}/BasePVObject.hpp
   ${SUBDIR}/buildandrun.hpp
   ${SUBDIR}/Communicator.hpp
   ${SUBDIR}/DataStore.hpp
   ${SUBDIR}/GaussianRandom.hpp
   ${SUBDIR}/HyPerCol.hpp
   ${SUBDIR}/InterColComm.hpp
   ${SUBDIR}/PV_Arguments.hpp
   ${SUBDIR}/PV_Factory.hpp
   ${SUBDIR}/PV_Init.hpp
   ${SUBDIR}/PV_KeywordHandler.hpp
   ${SUBDIR}/Random.hpp
)

