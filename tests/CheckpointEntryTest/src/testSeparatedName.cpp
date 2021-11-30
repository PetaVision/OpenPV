#include "testSeparatedName.hpp"
#include "checkpointing/CheckpointEntryData.hpp"
#include "checkpointing/CheckpointEntryPvpBuffer.hpp"
#include "include/PVLayerLoc.h"
#include "utils/PVLog.hpp"
#include <memory>

void testSeparatedName(std::shared_ptr<PV::MPIBlock const> mpiBlock) {
   std::string const correctName("separated_name");

   PV::CheckpointEntryData<float> separatedNameEntryData{
         "separated", "name", mpiBlock, (float *)nullptr, (size_t)0, false /*no broadcast*/};

   std::string const &entryDataName = separatedNameEntryData.getName();
   FatalIf(
         entryDataName != correctName,
         "testSeparatedName failed: name was \"%s\" instead of \"%s\".\n",
         entryDataName.c_str(),
         correctName.c_str());

   PV::CheckpointEntryPvpBuffer<float> separatedNameEntryPvp{"separated",
                                                             "name",
                                                             mpiBlock,
                                                             (float *)nullptr,
                                                             (PVLayerLoc const *)nullptr,
                                                             false /*no broadcast*/};

   std::string const &entryPvpName = separatedNameEntryPvp.getName();
   FatalIf(
         entryPvpName != correctName,
         "testSeparatedName failed: name was \"%s\" instead of \"%s\".\n",
         entryPvpName.c_str(),
         correctName.c_str());

   InfoLog() << "testSeparatedName passed.\n";
}
