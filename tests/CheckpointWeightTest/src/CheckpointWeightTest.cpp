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

PVLayerLoc setLayerLoc(
      PV::Communicator *communicator,
      int nBatchGlobal,
      int nxGlobal,
      int nyGlobal,
      int nf,
      int lt,
      int rt,
      int dn,
      int up);

PV::MPIBlock const setMPIBlock(PV::PV_Init &pv_initObj);

void verifyCheckpointing(PV::Weights &weights, PV::MPIBlock const mpiBlock);

int calcGlobalPatchIndex(
      int localPatchIndex,
      PV::MPIBlock const &mpiBlock,
      PVLayerLoc const &preLoc,
      PVLayerLoc const &postLoc,
      int nxp,
      int nyp);
float calcWeight(int patchIndex, int itemIndex, int numItemsInPatch);

bool isActiveWeight(PV::Patch const &patch, int nxp, int nyp, int nfp, int itemIndex);

int main(int argc, char *argv[]) {
   PV::PV_Init pv_initObj{&argc, &argv, false /*do not allow unrecognized arguments*/};
   PV::Communicator *communicator = pv_initObj.getCommunicator();
   PV::MPIBlock const mpiBlock    = setMPIBlock(pv_initObj);

   int const nxGlobal = 32;
   int const nyGlobal = 32;
   int const nfPre    = 3;
   int const nfPost   = 8;
   int const nxp      = 7;
   int const nyp      = 7;

   PVLayerLoc preLoc  = setLayerLoc(communicator, 1, nxGlobal, nyGlobal, nfPre, 3, 3, 3, 3);
   PVLayerLoc postLoc = setLayerLoc(communicator, 1, nxGlobal, nyGlobal, nfPost, 0, 0, 0, 0);

   PV::Weights weightsShared(
         std::string("shared"),
         nxp,
         nyp,
         nfPost,
         &preLoc,
         &postLoc,
         1 /* numArbors */,
         true /*sharedWeights */,
         0.0 /*timestamp*/);

   verifyCheckpointing(weightsShared, mpiBlock);

   PV::Weights weightsNonshared(
         std::string("nonshared"),
         nxp,
         nyp,
         nfPost,
         &preLoc,
         &postLoc,
         1 /* numArbors */,
         false /*sharedWeights */,
         0.0 /*timestamp*/);

   verifyCheckpointing(weightsNonshared, mpiBlock);

   return EXIT_SUCCESS;
}

void verifyCheckpointing(PV::Weights &weights, PV::MPIBlock const mpiBlock) {
   // Generate the weight data.
   // The weight value is patchIndex + weightIndex/(nxp*nyp*nfp), where
   // patchIndex is the index of the patch in global coordinates,
   // and weightIndex is the index of the location in the patch (in the range
   // 0 to nxp*nyp*nfp-1).
   weights.allocateDataStructures();
   int const nxp             = weights.getPatchSizeX();
   int const nyp             = weights.getPatchSizeY();
   int const nfp             = weights.getPatchSizeF();
   int const numItemsInPatch = nxp * nyp * nfp;
   bool const shared         = weights.getSharedFlag();
   int const numArbors       = weights.getNumArbors();
   int const numDataPatches  = weights.getNumDataPatches();
   PVLayerLoc const &preLoc  = weights.getGeometry()->getPreLoc();
   PVLayerLoc const &postLoc = weights.getGeometry()->getPostLoc();
   for (int a = 0; a < numArbors; a++) {
      float *arborDataStart = weights.getData(a);
      for (int p = 0; p < numDataPatches; p++) {
         int globalPatchIndex;
         if (shared) {
            globalPatchIndex = p;
         }
         else {
            globalPatchIndex = calcGlobalPatchIndex(p, mpiBlock, preLoc, postLoc, nxp, nyp);
         }
         for (int k = 0; k < numItemsInPatch; k++) {
            int const indexIntoArbor       = p * numItemsInPatch + k;
            float v                        = calcWeight(globalPatchIndex, k, numItemsInPatch);
            arborDataStart[indexIntoArbor] = v;
         }
      }
   }

   // For nonshared weights, set any weights outside shrunken patches to -1.
   // This allows us to check that, when a patch is split among more than one
   // process, values outside a shrunken patch on one process are not clobbeing
   // values inside a shrunken patch on another.
   if (!shared) {
      for (int a = 0; a < numArbors; a++) {
         float *arborDataStart = weights.getData(a);
         for (int p = 0; p < numDataPatches; p++) {
            PV::Patch const &patch = weights.getPatch(p);
            float *w               = weights.getDataFromDataIndex(a, p);
            for (int k = 0; k < numItemsInPatch; k++) {
               if (!isActiveWeight(patch, nxp, nyp, nfp, k)) {
                  w[k] = -1.0f;
               }
            }
         }
      }
   }

   // Delete checkpointDirectory and create it, if present; and then create an
   // empty checkpoint directory, to start fresh.
   std::string const checkpointDirectory("checkpoint");

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
   bool const compressFlag = false;
   auto checkpointEntry    = std::make_shared<PV::CheckpointEntryWeightPvp>(
         weights.getName(), &mpiBlock, &weights, compressFlag);

   double const timestamp = 10.0;
   bool verifyWritesFlag  = false;
   checkpointEntry->write(checkpointDirectory.c_str(), timestamp, verifyWritesFlag);

   // Overwrite the data
   for (int a = 0; a < numArbors; a++) {
      float *w            = weights.getData(a);
      int const arborSize = weights.getNumDataPatches() * numItemsInPatch;
      for (std::size_t d = 0; d < arborSize; d++) {
         w[d] = std::numeric_limits<float>::infinity();
      }
   }

   // Read the data back and verify timestamp.
   double readTime = (double)(timestamp == 0.0);
   checkpointEntry->read(checkpointDirectory.c_str(), &readTime);
   FatalIf(
         readTime != timestamp,
         "Timestamp read from checkpoint was %f; expected %f\n",
         readTime,
         timestamp);

   // Compare the weight data
   for (int a = 0; a < numArbors; a++) {
      float *arborDataStart = weights.getData(a);
      for (int p = 0; p < numDataPatches; p++) {
         int globalPatchIndex;
         if (shared) {
            globalPatchIndex = p;
         }
         else {
            globalPatchIndex = calcGlobalPatchIndex(p, mpiBlock, preLoc, postLoc, nxp, nyp);
         }
         PV::Patch const &patch = weights.getPatch(p);
         for (int k = 0; k < numItemsInPatch; k++) {
            if (shared or isActiveWeight(patch, nxp, nyp, nfp, k)) {
               int const indexIntoArbor = p * numItemsInPatch + k;
               float v                  = calcWeight(globalPatchIndex, k, numItemsInPatch);
               FatalIf(
                     arborDataStart[indexIntoArbor] != v,
                     "Rank %d, patch %d (global %d), patch item %d: expected %f; observed %f\n",
                     mpiBlock.getGlobalRank(),
                     p,
                     globalPatchIndex,
                     k,
                     (double)v,
                     (double)arborDataStart[indexIntoArbor]);
            }
         }
      }
   }

   if (mpiBlock.getRank() == 0) {
      InfoLog().printf("%s passed.\n", weights.getName().c_str());
   }
}

PVLayerLoc setLayerLoc(
      PV::Communicator *communicator,
      int nBatchGlobal,
      int nxGlobal,
      int nyGlobal,
      int nf,
      int lt,
      int rt,
      int dn,
      int up) {
   int nBatchLocal = nBatchGlobal / communicator->numCommBatches();
   FatalIf(
         nBatchLocal * communicator->numCommBatches() != nBatchGlobal,
         "Number of MPI batch elements %d does not divide nBatchGlobal %d\n",
         communicator->numCommBatches(),
         nBatchGlobal);
   int nxLocal = nxGlobal / communicator->numCommColumns();
   FatalIf(
         nxLocal * communicator->numCommColumns() != nxGlobal,
         "Number of MPI columns %d does not divide nxGlobal %d\n",
         communicator->numCommColumns(),
         nxGlobal);
   int nyLocal = nyGlobal / communicator->numCommRows();
   FatalIf(
         nyLocal * communicator->numCommRows() != nyGlobal,
         "Number of MPI rows %d does not divide nyGlobal %d\n",
         communicator->numCommRows(),
         nyGlobal);

   PVLayerLoc layerLoc;
   layerLoc.nbatch       = nBatchLocal;
   layerLoc.nx           = nxLocal;
   layerLoc.ny           = nyLocal;
   layerLoc.nf           = nf;
   layerLoc.nbatchGlobal = nBatchGlobal;
   layerLoc.nxGlobal     = nxGlobal;
   layerLoc.nyGlobal     = nyGlobal;
   layerLoc.kb0          = nBatchLocal * communicator->commBatch();
   layerLoc.kx0          = nxLocal * communicator->commColumn();
   layerLoc.ky0          = nyLocal * communicator->commRow();
   layerLoc.halo.lt      = lt;
   layerLoc.halo.rt      = rt;
   layerLoc.halo.dn      = dn;
   layerLoc.halo.up      = up;

   return layerLoc;
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

PV::MPIBlock const setMPIBlock(PV::PV_Init &pv_initObj) {
   PV::Arguments const *arguments     = pv_initObj.getArguments();
   PV::MPIBlock const *globalMPIBlock = pv_initObj.getCommunicator()->getGlobalMPIBlock();
   PV::Checkpointer tempCheckpointer("column", globalMPIBlock, arguments);
   PV::MPIBlock const mpiBlock = *tempCheckpointer.getMPIBlock();
   return mpiBlock;
}

float calcWeight(int patchIndex, int itemIndex, int numItemsInPatch) {
   return (float)patchIndex + (float)itemIndex / (float)numItemsInPatch;
}

bool isActiveWeight(PV::Patch const &patch, int nxp, int nyp, int nfp, int itemIndex) {
   int const x = kxPos(itemIndex, nxp, nyp, nfp);
   int const y = kyPos(itemIndex, nxp, nyp, nfp);

   int const startx = kxPos(patch.offset, nxp, nyp, nfp);
   int const starty = kyPos(patch.offset, nxp, nyp, nfp);

   return (x >= startx and x < startx + patch.nx and y >= starty and y < starty + patch.ny);
}
