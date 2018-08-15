set (PVLibSrcCpp ${PVLibSrcCpp}
   ${SUBDIR}/ANNWhitenedLayer.cpp
   ${SUBDIR}/ImageFromMemoryBuffer.cpp
   ${SUBDIR}/KmeansLayer.cpp
   ${SUBDIR}/LabelErrorLayer.cpp
   ${SUBDIR}/LCALIFLayer.cpp
   ${SUBDIR}/RunningAverageLayer.cpp
   ${SUBDIR}/ShuffleLayer.cpp
)

set (PVLibSrcHpp ${PVLibSrcHpp}
   ${SUBDIR}/ANNWhitenedLayer.hpp
   ${SUBDIR}/ImageFromMemoryBuffer.hpp
   ${SUBDIR}/KmeansLayer.hpp
   ${SUBDIR}/LabelErrorLayer.hpp
   ${SUBDIR}/LCALIFLayer.hpp
   ${SUBDIR}/RunningAverageLayer.hpp
   ${SUBDIR}/ShuffleLayer.hpp
)

set (PVLibSrcH ${PVLibSrcH}
   ${SUBDIR}/DeprecatedUpdateStateFunctions.h
)
