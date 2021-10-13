set (PVLibSrcCpp ${PVLibSrcCpp}
#   ${SUBDIR}/ANNWhitenedLayer.cpp
#   ${SUBDIR}/ImageFromMemoryBuffer.cpp
#   ${SUBDIR}/KmeansLayer.cpp
#   ${SUBDIR}/LabelErrorLayer.cpp
#   ${SUBDIR}/LCALIFLayer.cpp
#   ${SUBDIR}/RunningAverageLayer.cpp
#   ${SUBDIR}/ShuffleLayer.cpp
#   ${SUBDIR}/WTALayer.cpp
    ${SUBDIR}/BaseConnectionProbe.cpp
    ${SUBDIR}/BaseHyPerConnProbe.cpp
    ${SUBDIR}/KernelProbe.cpp
    ${SUBDIR}/L2ConnProbe.cpp
    ${SUBDIR}/PointProbe.cpp
    ${SUBDIR}/PointLIFProbe.cpp
)

set (PVLibSrcHpp ${PVLibSrcHpp}
#   ${SUBDIR}/ANNWhitenedLayer.hpp
#   ${SUBDIR}/ImageFromMemoryBuffer.hpp
#   ${SUBDIR}/KmeansLayer.hpp
#   ${SUBDIR}/LabelErrorLayer.hpp
#   ${SUBDIR}/LCALIFLayer.hpp
#   ${SUBDIR}/RunningAverageLayer.hpp
#   ${SUBDIR}/ShuffleLayer.hpp
#   ${SUBDIR}/WTALayer.hpp
    ${SUBDIR}/BaseConnectionProbe.hpp
    ${SUBDIR}/BaseHyPerConnProbe.hpp
    ${SUBDIR}/KernelProbe.hpp
    ${SUBDIR}/L2ConnProbe.hpp
    ${SUBDIR}/PointProbe.hpp
    ${SUBDIR}/PointLIFProbe.hpp
)

#Deprecated Oct 7, 2021.
#The probe classes listed here currently are not being used, and
#maintaining them would complicate refactoring.

set (PVLibSrcH ${PVLibSrcH}
#   ${SUBDIR}/DeprecatedUpdateStateFunctions.h
#   ${SUBDIR}/updateStateFunctions.h
)
