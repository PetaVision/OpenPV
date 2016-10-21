/*
 * pv.cpp
 *
 */

#include "checkpointing/CheckpointEntryWeightPvp.hpp"
#include "columns/HyPerCol.hpp"
#include "columns/PV_Init.hpp"
#include "connections/HyPerConn.hpp"
#include <cstdlib>
#include <memory>
#include <vector>

int main(int argc, char *argv[]) {
   std::string const checkpointDirectory("checkpoint");
   std::string const paramsFile("input/CheckpointWeightTest.params");
   std::string const connectionName("InputToOutput");

   // Delete checkpointDirectory, if present, to start fresh
   std::string rmrfcommand("rm -rf ");
   rmrfcommand.append(checkpointDirectory);
   system(rmrfcommand.c_str());

   // Create and run the column, and then retrieve the connection
   PV::PV_Init pv_initObj{&argc, &argv, false /*do not allow unrecognized arguments*/};
   PV::HyPerCol *hc = createHyPerCol(&pv_initObj);
   pvErrorIf(hc == nullptr, "Failed to create HyPerCol.\n");
   hc->run();
   PV::HyPerConn *conn = dynamic_cast<PV::HyPerConn *>(hc->getConnFromName(connectionName.c_str()));
   pvErrorIf(
         conn == nullptr, "No connection named % in %s\n", connectionName.c_str(), hc->getName());

   int const numArbors      = conn->numberOfAxonalArborLists();
   int const patchDataSize  = conn->getNumWeightPatches();
   int const weightDataSize = conn->getNumDataPatches();
   int const nxp            = conn->xPatchSize();
   int const nyp            = conn->yPatchSize();
   int const nfp            = conn->fPatchSize();

   // Create CheckpointEntryWeightPvp object.
   // We will copy the patch and weight data to external buffers, create the CheckpointEntry,
   // write the checkpoint, then overwrite the data, and finally read the checkpoint back
   // and compare with the original.

   // Recreate the patch data.
   std::vector<PVPatch **> patches{(std::size_t)numArbors};
   for (auto &p : patches) {
      p = PV::HyPerConn::createPatches(patchDataSize, nxp, nyp);
   }

   // Copy the weight data
   std::vector<std::vector<pvdata_t>> weights{(std::size_t)numArbors};
   for (int a = 0; a < numArbors; a++) {
      std::vector<pvdata_t> &w = weights.at(a);
      w.resize(weightDataSize * nxp * nyp * nfp);
      pvdata_t *destWeights         = w.data();
      pvdata_t const *sourceWeights = conn->get_wDataStart(a);
      memcpy(destWeights, sourceWeights, sizeof(pvdata_t) * weightDataSize * nxp * nyp * nfp);
   }
   std::vector<pvdata_t *> weightPointers{(std::size_t)numArbors};
   for (int a = 0; a < numArbors; a++) {
      weightPointers[a] = weights[a].data();
   }

   // Create the CheckpointEntry.
   auto checkpointEntry = std::make_shared<PV::CheckpointEntryWeightPvp>(
         std::string(connectionName.c_str()),
         pv_initObj.getCommunicator(),
         numArbors,
         conn->usingSharedWeights(),
         patches.data(),
         patchDataSize,
         weightPointers.data(),
         weightDataSize,
         nxp,
         nyp,
         nfp,
         conn->preSynapticLayer()->getLayerLoc(),
         conn->postSynapticLayer()->getLayerLoc(),
         false /*do not compress*/);

   PV::ensureDirExists(pv_initObj.getCommunicator(), checkpointDirectory.c_str());
   checkpointEntry->write(checkpointDirectory.c_str(), hc->simulationTime(), false);

   // Overwrite the data
   for (int a = 0; a < numArbors; a++) {
      std::vector<pvdata_t> &w = weights.at(a);
      for (std::size_t d = 0; d < w.size(); d++) {
         w[d] = (w[d] == (pvdata_t)d) ? (pvdata_t)-1.0 : (pvdata_t)d;
      }
   }

   double readTime = 0.0;
   checkpointEntry->read(checkpointDirectory.c_str(), &readTime);

   // Compare the weight data
   for (int a = 0; a < numArbors; a++) {
      std::vector<pvdata_t> &w      = weights.at(a);
      pvdata_t *destWeights         = w.data();
      pvdata_t const *sourceWeights = conn->get_wDataStart(a);
      for (int k = 0; k < weightDataSize * nxp * nyp * nfp; k++) {
         pvErrorIf(destWeights[k] != sourceWeights[k], "%s failed.\n", argv[0]);
      }
   }

   delete hc;
   return PV_SUCCESS;
}
