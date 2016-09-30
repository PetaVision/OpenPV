#include "testSeparatedName.hpp"
#include "io/CheckpointEntry.hpp"
#include "utils/PVLog.hpp"

void testSeparatedName(PV::Communicator *comm) {
   std::string const correctName("separated_name");

   PV::CheckpointEntryData<float> separatedNameEntryData{
         "separated", "name", comm, (float *)nullptr, (size_t)0, false /*no broadcast*/};

   std::string const &entryDataName = separatedNameEntryData.getName();
   pvErrorIf(
         entryDataName != correctName,
         "testSeparatedName failed: name was \"%s\" instead of \"%s\".\n",
         entryDataName.c_str(),
         correctName.c_str());

   PV::CheckpointEntryPvp<float> separatedNameEntryPvp{"separated",
                                                       "name",
                                                       comm,
                                                       (float *)nullptr,
                                                       (size_t)0,
                                                       PV_FLOAT_TYPE,
                                                       (PVLayerLoc const *)nullptr,
                                                       false /*no broadcast*/};

   std::string const &entryPvpName = separatedNameEntryPvp.getName();
   pvErrorIf(
         entryPvpName != correctName,
         "testSeparatedName failed: name was \"%s\" instead of \"%s\".\n",
         entryPvpName.c_str(),
         correctName.c_str());

   pvInfo() << "testSeparatedName passed.\n";
}
