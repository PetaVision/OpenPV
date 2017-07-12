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

int calcGlobalPatchIndex(
      int localPatchIndex,
      PV::MPIBlock const &mpiBlock,
      PVLayerLoc const &preLoc,
      PVLayerLoc const &postLoc,
      int nxp,
      int nyp);
float calcWeight(int patchIndex, int itemIndex, int numItemsInPatch);
bool isActiveWeight(
      int itemIndex,
      int localPatchIndex,
      PVPatch const *const *arborPatches,
      int nxp,
      int nyp,
      int nfp);

int main(int argc, char *argv[]) {
   std::string const checkpointDirectory("checkpoint");
   std::string const paramsFile("input/CheckpointWeightTest.params");
   std::string const connectionName("InputToOutput");

   PV::PV_Init pv_initObj{&argc, &argv, false /*do not allow unrecognized arguments*/};

   // Create the column, and then retrieve the connection
   // We need the number patches and the patch size, but will generate our
   // own weights, for easier diagnosis.
   PV::HyPerCol *hc = new PV::HyPerCol(&pv_initObj);
   FatalIf(hc == nullptr, "Failed to create HyPerCol.\n");
   hc->allocateColumn();
   PV::HyPerConn *conn = dynamic_cast<PV::HyPerConn *>(hc->getObjectFromName(connectionName));
   FatalIf(conn == nullptr, "No connection named % in %s\n", connectionName.c_str(), hc->getName());

   double const timestamp    = hc->getStopTime();
   int const numArbors       = conn->numberOfAxonalArborLists();
   int const shared          = conn->usingSharedWeights();
   int const patchDataSize   = conn->getNumWeightPatches();
   int const numDataPatchesX = conn->getNumDataPatchesX();
   int const numDataPatchesY = conn->getNumDataPatchesY();
   int const numDataPatchesF = conn->getNumDataPatchesF();
   int const numDataPatches  = conn->getNumDataPatches();
   pvAssert(numDataPatches == numDataPatchesX * numDataPatchesY * numDataPatchesF);

   int const nxp = conn->xPatchSize();
   int const nyp = conn->yPatchSize();
   int const nfp = conn->fPatchSize();

   PVLayerLoc const preLoc  = *conn->preSynapticLayer()->getLayerLoc();
   PVLayerLoc const postLoc = *conn->postSynapticLayer()->getLayerLoc();

   PV::Arguments const *arguments     = pv_initObj.getArguments();
   PV::MPIBlock const *globalMPIBlock = pv_initObj.getCommunicator()->getGlobalMPIBlock();
   PV::Checkpointer *tempCheckpointer = new PV::Checkpointer("column", globalMPIBlock, arguments);
   PV::MPIBlock const mpiBlock        = *tempCheckpointer->getMPIBlock();
   delete tempCheckpointer;

   // Copy over the PVPatch information, and delete the column.
   std::vector<PVPatch **> patches{(std::size_t)numArbors};
   for (int arbor = 0; arbor < numArbors; arbor++) {
      patches[arbor] = PV::HyPerConn::createPatches(patchDataSize, nxp, nyp);
      for (int patchIndex = 0; patchIndex < numDataPatches; patchIndex++) {
         memcpy(patches[arbor][patchIndex], conn->getWeights(patchIndex, arbor), sizeof(PVPatch));
      }
   }
   delete hc;

   // Generate the weight data. For easier diagnosis of problems, we
   // do not use the weights specified in the params file, but use a formula
   // to generate unique weights for each independent connection.
   // The weight value is patchIndex + weightIndex/(nxp*nyp*nfp), where
   // patchIndex is the index of the patch in global coordinates,
   // and weightIndex is the index of the location in the patch (in the range
   // 0 to nxp*nyp*nfp-1).
   std::vector<std::vector<float>> weights{(std::size_t)numArbors};
   int const numItemsInPatch = nxp * nyp * nfp;
   for (int a = 0; a < numArbors; a++) {
      std::vector<float> &w = weights.at(a);
      w.resize(numDataPatches * numItemsInPatch);
      for (int p = 0; p < numDataPatches; p++) {
         int globalPatchIndex;
         if (shared) {
            globalPatchIndex = p;
         }
         else {
            globalPatchIndex = calcGlobalPatchIndex(p, mpiBlock, preLoc, postLoc, nxp, nyp);
         }
         for (int k = 0; k < numItemsInPatch; k++) {
            int const indexIntoArbor = p * numItemsInPatch + k;
            float v                  = calcWeight(globalPatchIndex, k, numItemsInPatch);
            w.at(indexIntoArbor)     = v;
         }
      }
   }
   std::vector<float *> weightPointers{(std::size_t)numArbors};
   for (int a = 0; a < numArbors; a++) {
      weightPointers[a] = weights[a].data();
   }

   // For nonshared weights, set any weights outside shrunken patches to -1.
   // This allows us to check that, when a patch is split among more than one
   // process, values outside a shrunken patch on one process are not clobbeing
   // values inside a shrunken patch on another.
   if (!shared) {
      for (int a = 0; a < numArbors; a++) {
         PVPatch const *const *arborPatches = patches.at(a);
         std::vector<float> &arborWeights   = weights.at(a);
         for (int p = 0; p < numDataPatches; p++) {
            float *w = &arborWeights.at(p * numItemsInPatch);
            for (int k = 0; k < numItemsInPatch; k++) {
               if (!isActiveWeight(k, p, arborPatches, nxp, nyp, nfp)) {
                  w[k] = -1.0f;
               }
            }
         }
      }
   }

   // Delete checkpointDirectory and create it, if present; and then create an
   // empty checkpoint directory, to start fresh.
   int globalSize;
   MPI_Comm_size(mpiBlock.getGlobalComm(), &globalSize);
   for (int globalRank = 0; globalRank < globalSize; globalRank++) {
      if (mpiBlock.getGlobalRank() == globalRank and mpiBlock.getRank() == 0) {
         struct stat checkpointStat;
         int statStatus = stat(checkpointDirectory.c_str(), &checkpointStat);
         if (statStatus == -1) {
            if (errno == ENOENT) {
               continue;
            }
            else {
               Fatal().printf(
                     "Failed checking status of %s: %s\n",
                     checkpointDirectory.c_str(),
                     strerror(errno));
            }
         }
         std::string rmrfcommand("rm -rf ");
         rmrfcommand.append(checkpointDirectory);
         int status = system(rmrfcommand.c_str());
         FatalIf(status, "Failed to delete %s\n", checkpointDirectory.c_str());
      }
      MPI_Barrier(mpiBlock.getGlobalComm());
   }
   PV::ensureDirExists(&mpiBlock, checkpointDirectory.c_str());

   // Create the CheckpointEntry.
   auto checkpointEntry = std::make_shared<PV::CheckpointEntryWeightPvp>(
         connectionName,
         &mpiBlock,
         numArbors,
         shared,
         patches.data(),
         weightPointers.data(),
         numDataPatchesX,
         numDataPatchesY,
         numDataPatchesF,
         nxp,
         nyp,
         nfp,
         &preLoc,
         &postLoc,
         false /*do not compress*/);

   checkpointEntry->write(checkpointDirectory.c_str(), timestamp, false);

   // Overwrite the data
   for (int a = 0; a < numArbors; a++) {
      std::vector<float> &w = weights.at(a);
      for (std::size_t d = 0; d < w.size(); d++) {
         w[d] = std::numeric_limits<float>::infinity();
      }
   }

   // Read the data back
   double readTime = (double)(timestamp != 0.0);
   checkpointEntry->read(checkpointDirectory.c_str(), &readTime);

   // Compare the weight data
   for (int a = 0; a < numArbors; a++) {
      PVPatch const *const *arborPatches = patches.at(a);
      std::vector<float> &arborWeights   = weights.at(a);
      for (int p = 0; p < numDataPatches; p++) {
         int globalPatchIndex;
         if (shared) {
            globalPatchIndex = p;
         }
         else {
            globalPatchIndex = calcGlobalPatchIndex(p, mpiBlock, preLoc, postLoc, nxp, nyp);
         }
         for (int k = 0; k < numItemsInPatch; k++) {
            if (shared or isActiveWeight(k, p, arborPatches, nxp, nyp, nfp)) {
               int const indexIntoArbor = p * numItemsInPatch + k;
               float v                  = calcWeight(globalPatchIndex, k, numItemsInPatch);
               FatalIf(
                     arborWeights.at(indexIntoArbor) != v,
                     "Rank %d, patch %d (global %d), patch item %d: expected %f; observed %f\n",
                     mpiBlock.getGlobalRank(),
                     p,
                     globalPatchIndex,
                     k,
                     (double)v,
                     (double)arborWeights.at(indexIntoArbor));
            }
         }
      }
   }

   if (mpiBlock.getRank() == 0) {
      InfoLog() << "Test passed." << std::endl;
   }
   return EXIT_SUCCESS;
}

int calcGlobalPatchIndex(
      int localPatchIndex,
      PV::MPIBlock const &mpiBlock,
      PVLayerLoc const &preLoc,
      PVLayerLoc const &postLoc,
      int nxp,
      int nyp) {
   int marginX           = requiredConvolveMargin(preLoc.nx, postLoc.nx, nxp);
   int marginY           = requiredConvolveMargin(preLoc.ny, postLoc.ny, nyp);
   int numPatchesX       = preLoc.nx + marginX + marginX;
   int numPatchesY       = preLoc.ny + marginY + marginY;
   int const nf          = preLoc.nf;
   int numPatchesXGlobal = preLoc.nxGlobal + marginX + marginX;
   int numPatchesYGlobal = preLoc.nyGlobal + marginY + marginY;

   int x = kxPos(localPatchIndex, numPatchesX, numPatchesY, nf);
   x += preLoc.nx * (mpiBlock.getStartColumn() + mpiBlock.getColumnIndex());
   int y = kyPos(localPatchIndex, numPatchesX, numPatchesY, nf);
   y += preLoc.ny * (mpiBlock.getStartRow() + mpiBlock.getRowIndex());
   int f = featureIndex(localPatchIndex, numPatchesX, numPatchesY, nf);

   int patchIndexGlobal = kIndex(x, y, f, numPatchesXGlobal, numPatchesYGlobal, nf);
   return patchIndexGlobal;
}

float calcWeight(int patchIndex, int itemIndex, int numItemsInPatch) {
   return (float)patchIndex + (float)itemIndex / (float)numItemsInPatch;
}

bool isActiveWeight(
      int itemIndex,
      int localPatchIndex,
      PVPatch const *const *arborPatches,
      int nxp,
      int nyp,
      int nfp) {
   int const x = kxPos(itemIndex, nxp, nyp, nfp);
   int const y = kyPos(itemIndex, nxp, nyp, nfp);

   PVPatch const *patch = arborPatches[localPatchIndex];
   int const startx     = kxPos(patch->offset, nxp, nyp, nfp);
   int const starty     = kyPos(patch->offset, nxp, nyp, nfp);

   return (x >= startx and x < startx + patch->nx and y >= starty and y < starty + patch->ny);
}
