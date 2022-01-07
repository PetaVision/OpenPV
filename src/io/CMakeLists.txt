set (PVLibSrcCpp ${PVLibSrcCpp}
   ${SUBDIR}/ConfigParser.cpp
   ${SUBDIR}/Configuration.cpp
   ${SUBDIR}/fileio.cpp
   ${SUBDIR}/FileManager.cpp
   ${SUBDIR}/FileStream.cpp
   ${SUBDIR}/io.cpp
   ${SUBDIR}/MPIRecvStream.cpp
   ${SUBDIR}/MPISendStream.cpp
   ${SUBDIR}/PVParams.cpp
   ${SUBDIR}/randomstateio.cpp
   ${SUBDIR}/WeightsFileIO.cpp
)

set (PVLibSrcHpp ${PVLibSrcHpp}
   ${SUBDIR}/ConfigParser.hpp
   ${SUBDIR}/Configuration.hpp
   ${SUBDIR}/fileio.hpp
   ${SUBDIR}/FileManager.hpp
   ${SUBDIR}/FileStream.hpp
   ${SUBDIR}/io.hpp
   ${SUBDIR}/MPIRecvStream.hpp
   ${SUBDIR}/MPISendStream.hpp
   ${SUBDIR}/PrintStream.hpp
   ${SUBDIR}/PVParams.hpp
   ${SUBDIR}/randomstateio.hpp
   ${SUBDIR}/WeightsFileIO.hpp
)
