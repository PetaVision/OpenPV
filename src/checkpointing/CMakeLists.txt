set (PVLibSrcCpp ${PVLibSrcCpp}
   ${SUBDIR}/CheckpointEntry.cpp
   ${SUBDIR}/CheckpointEntryDataStore.cpp
   ${SUBDIR}/CheckpointEntryFilePosition.cpp
   ${SUBDIR}/CheckpointEntryParamsFileWriter.cpp
   ${SUBDIR}/CheckpointEntryRandState.cpp
   ${SUBDIR}/CheckpointEntryWeightPvp.cpp
   ${SUBDIR}/Checkpointer.cpp
   ${SUBDIR}/CheckpointerDataInterface.cpp
)

set (PVLibSrcHpp ${PVLibSrcHpp}
   ${SUBDIR}/CheckpointEntry.hpp
   ${SUBDIR}/CheckpointEntryData.hpp
   ${SUBDIR}/CheckpointEntryDataStore.hpp
   ${SUBDIR}/CheckpointEntryFilePosition.hpp
   ${SUBDIR}/CheckpointEntryParamsFileWriter.hpp
   ${SUBDIR}/CheckpointEntryPvp.hpp
   ${SUBDIR}/CheckpointEntryPvpBuffer.hpp
   ${SUBDIR}/CheckpointEntryRandState.hpp
   ${SUBDIR}/CheckpointEntryWeightPvp.hpp
   ${SUBDIR}/Checkpointer.hpp
   ${SUBDIR}/CheckpointerDataInterface.hpp
   ${SUBDIR}/CheckpointingMessages.hpp
)

set (PVLibSrcTpp ${PVLibSrcTpp}
   ${SUBDIR}/CheckpointEntryData.tpp
   ${SUBDIR}/CheckpointEntryPvp.tpp
   ${SUBDIR}/CheckpointEntryPvpBuffer.tpp
)

if (PV_USE_MPI)
   set (PVLibSrcCpp ${PVLibSrcCpp} ${SUBDIR}/CheckpointEntryMPIRecvStream.cpp)
   set (PVLibSrcHpp ${PVLibSrcHpp} ${SUBDIR}/CheckpointEntryMPIRecvStream.hpp)
endif()
