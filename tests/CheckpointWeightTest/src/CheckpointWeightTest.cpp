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

   PV::PV_Init pv_initObj{&argc, &argv, false /*do not allow unrecognized arguments*/};

   // Delete checkpointDirectory, if present, to start fresh
   if (pv_initObj.getCommunicator()->commRank() == 0) {
      std::string rmrfcommand("rm -rf ");
      rmrfcommand.append(checkpointDirectory);
      int status = system(rmrfcommand.c_str());
      FatalIf(status, "%s failed to delete %s\n", argv[0], checkpointDirectory.c_str());
   }

   // Create and run the column, and then retrieve the connection
   PV::HyPerCol *hc = createHyPerCol(&pv_initObj);
   FatalIf(hc == nullptr, "Failed to create HyPerCol.\n");
   hc->run();
   PV::HyPerConn *conn = dynamic_cast<PV::HyPerConn *>(hc->getConnFromName(connectionName.c_str()));
   FatalIf(conn == nullptr, "No connection named % in %s\n", connectionName.c_str(), hc->getName());

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
   std::vector<std::vector<float>> weights{(std::size_t)numArbors};
   for (int a = 0; a < numArbors; a++) {
      std::vector<float> &w = weights.at(a);
      w.resize(weightDataSize * nxp * nyp * nfp);
      float *destWeights         = w.data();
      float const *sourceWeights = conn->get_wDataStart(a);
      memcpy(destWeights, sourceWeights, sizeof(float) * weightDataSize * nxp * nyp * nfp);
   }
   std::vector<float *> weightPointers{(std::size_t)numArbors};
   for (int a = 0; a < numArbors; a++) {
      weightPointers[a] = weights[a].data();
   }

   PV::MPIBlock const *mpiBlock = pv_initObj.getCommunicator()->getLocalMPIBlock();
   // Create the CheckpointEntry.
   auto checkpointEntry = std::make_shared<PV::CheckpointEntryWeightPvp>(
         std::string(connectionName.c_str()),
         mpiBlock,
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

   PV::ensureDirExists(mpiBlock, checkpointDirectory.c_str());
   checkpointEntry->write(checkpointDirectory.c_str(), hc->simulationTime(), false);

   // Overwrite the data
   for (int a = 0; a < numArbors; a++) {
      std::vector<float> &w = weights.at(a);
      for (std::size_t d = 0; d < w.size(); d++) {
         w[d] = (w[d] == (float)d) ? (float)-1.0 : (float)d;
      }
   }

   double readTime = 0.0;
   checkpointEntry->read(checkpointDirectory.c_str(), &readTime);

   // Compare the weight data
   for (int a = 0; a < numArbors; a++) {
      std::vector<float> &w      = weights.at(a);
      float *destWeights         = w.data();
      float const *sourceWeights = conn->get_wDataStart(a);
      for (int k = 0; k < weightDataSize * nxp * nyp * nfp; k++) {
         FatalIf(destWeights[k] != sourceWeights[k], "%s failed.\n", argv[0]);
      }
   }

   delete hc;
   return PV_SUCCESS;
}
