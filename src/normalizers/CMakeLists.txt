set (PVLibSrcCpp ${PVLibSrcCpp}
   ${SUBDIR}/NormalizeBase.cpp
   ${SUBDIR}/NormalizeContrastZeroMean.cpp
   ${SUBDIR}/NormalizeGroup.cpp
   ${SUBDIR}/NormalizeL2.cpp
   ${SUBDIR}/NormalizeMax.cpp
   ${SUBDIR}/NormalizeMultiply.cpp
   ${SUBDIR}/NormalizeNone.cpp
   ${SUBDIR}/NormalizeSum.cpp
)

set (PVLibSrcHpp ${PVLibSrcHpp}
   ${SUBDIR}/NormalizeBase.hpp
   ${SUBDIR}/NormalizeContrastZeroMean.hpp
   ${SUBDIR}/NormalizeGroup.hpp
   ${SUBDIR}/NormalizeL2.hpp
   ${SUBDIR}/NormalizeMax.hpp
   ${SUBDIR}/NormalizeNone.hpp
   ${SUBDIR}/NormalizeSum.hpp
)

