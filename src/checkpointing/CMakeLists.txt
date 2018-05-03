set (PVLibSrcCpp ${PVLibSrcCpp}
   ${SUBDIR}/CheckpointEntry.cpp
   ${SUBDIR}/CheckpointEntryDataStore.cpp
   ${SUBDIR}/CheckpointEntryRandState.cpp
   ${SUBDIR}/CheckpointEntryWeightPvp.cpp
   ${SUBDIR}/CheckpointableFileStream.cpp
   ${SUBDIR}/Checkpointer.cpp
   ${SUBDIR}/CheckpointerDataInterface.cpp
)

set (PVLibSrcHpp ${PVLibSrcHpp}
   ${SUBDIR}/CheckpointEntry.hpp
   ${SUBDIR}/CheckpointEntryData.hpp
   ${SUBDIR}/CheckpointEntryDataStore.hpp
   ${SUBDIR}/CheckpointEntryPvp.hpp
   ${SUBDIR}/CheckpointEntryPvpBuffer.hpp
   ${SUBDIR}/CheckpointEntryRandState.hpp
   ${SUBDIR}/CheckpointEntryWeightPvp.hpp
   ${SUBDIR}/CheckpointableFileStream.hpp
   ${SUBDIR}/Checkpointer.hpp
   ${SUBDIR}/CheckpointerDataInterface.hpp
   ${SUBDIR}/CheckpointingMessages.hpp
)

set (PVLibSrcTpp ${PVLibSrcTpp}
   ${SUBDIR}/CheckpointEntryData.tpp
   ${SUBDIR}/CheckpointEntryPvp.tpp
   ${SUBDIR}/CheckpointEntryPvpBuffer.tpp
)
