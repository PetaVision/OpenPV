/*
 * pv.cpp
 *
 */

#include "checkpointing/CheckpointEntryWeightPvp.hpp"
#include "columns/HyPerCol.hpp"
#include "columns/PV_Init.hpp"
#include "connections/HyPerConn.hpp"
#include "io/FileManager.hpp"
#include "utils/requiredConvolveMargin.hpp"
#include <cstdlib>
#include <memory>
#include <vector>

using namespace PV;

PVLayerLoc setLayerLoc(
      Communicator const *communicator,
      int nBatchGlobal,
      int nxGlobal,
      int nyGlobal,
      int nf,
      int lt,
      int rt,
      int dn,
      int up);

MPIBlock const setMPIBlock(PV_Init &pv_initObj);

void verifyCheckpointing(
      int nxp, int nyp, int nfp,
      PVLayerLoc const &preLoc, PVLayerLoc const &postLoc,
      bool sharedFlag, std::shared_ptr<MPIBlock const> mpiBlock);

int calcGlobalPatchIndex(
      int localPatchIndex,
      std::shared_ptr<MPIBlock const> mpiBlock,
      PVLayerLoc const &preLoc,
      PVLayerLoc const &postLoc,
      int nxp,
      int nyp);
float calcWeight(int patchIndex, int itemIndex, int numItemsInPatch);

bool isActiveWeight(Patch const &patch, int nxp, int nyp, int nfp, int itemIndex);

int main(int argc, char *argv[]) {
   PV_Init pv_initObj{&argc, &argv, false /*do not allow unrecognized arguments*/};
   Communicator const *communicator = pv_initObj.getCommunicator();
   std::shared_ptr<MPIBlock const> mpiBlock = communicator->getIOMPIBlock();

   int const nxGlobal = 32;
   int const nyGlobal = 32;
   int const nfPre    = 3;
   int const nfPost   = 8;
   int const nxp      = 7;
   int const nyp      = 7;

   PVLayerLoc preLoc  = setLayerLoc(communicator, 1, nxGlobal, nyGlobal, nfPre, 3, 3, 3, 3);
   PVLayerLoc postLoc = setLayerLoc(communicator, 1, nxGlobal, nyGlobal, nfPost, 0, 0, 0, 0);

   verifyCheckpointing(nxp, nyp, nfPost, preLoc, postLoc, true /*sharedFlag*/, mpiBlock);

   verifyCheckpointing(nxp, nyp, nfPost, preLoc, postLoc, false /*sharedFlag*/, mpiBlock);

   return EXIT_SUCCESS;
}

void verifyCheckpointing(
      int nxp, int nyp, int nfp,
      PVLayerLoc const &preLoc, PVLayerLoc const &postLoc,
      bool sharedFlag, std::shared_ptr<MPIBlock const> mpiBlock) {
   // Create the weights
   std::string label(sharedFlag ? "shared" : "nonshared");
   int const numArbors = 1;
   Weights weights(
         label, nxp, nyp, nfp, &preLoc, &postLoc, numArbors, sharedFlag, 0.0 /*timestamp*/);

   // Generate the weight data.
   // The weight value is patchIndex + weightIndex/(nxp*nyp*nfp), where
   // patchIndex is the index of the patch in global coordinates,
   // and weightIndex is the index of the location in the patch (in the range
   // 0 to nxp*nyp*nfp-1).
   weights.allocateDataStructures();
   int const numItemsInPatch = nxp * nyp * nfp;
   int const numDataPatches  = weights.getNumDataPatches();
   for (int a = 0; a < numArbors; a++) {
      float *arborDataStart = weights.getData(a);
      for (int p = 0; p < numDataPatches; p++) {
         int globalPatchIndex;
         if (sharedFlag) {
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
   if (!sharedFlag) {
      for (int a = 0; a < numArbors; a++) {
         for (int p = 0; p < numDataPatches; p++) {
            Patch const &patch = weights.getPatch(p);
            float *w           = weights.getDataFromDataIndex(a, p);
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
   auto fileManager = std::make_shared<FileManager>(mpiBlock, checkpointDirectory);

   int globalSize;
   MPI_Comm_size(mpiBlock->getGlobalComm(), &globalSize);
   for (int globalRank = 0; globalRank < globalSize; globalRank++) {
      if (mpiBlock->getGlobalRank() == globalRank and mpiBlock->getRank() == 0) {
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
      MPI_Barrier(mpiBlock->getGlobalComm());
   }
   ensureDirExists(mpiBlock, checkpointDirectory.c_str());

   // Create the CheckpointEntry.
   bool const compressFlag = false;
   auto checkpointWriter   = std::make_shared<CheckpointEntryWeightPvp>(
         weights.getName(), &weights, compressFlag);

   double const timestamp = 10.0;
   bool verifyWritesFlag  = false;
   checkpointWriter->write(fileManager, timestamp, verifyWritesFlag);

   // Create a Weights object to read the checkpoint into
   Weights readBack(
         label, nxp, nyp, nfp, &preLoc, &postLoc, numArbors, sharedFlag, 0.0 /*timestamp*/);
   readBack.allocateDataStructures();
   // Initialize readBack values to infinity, to catch errors where checkpoint read does nothing.
   for (int a = 0; a < numArbors; a++) {
      float *w            = readBack.getData(a);
      int const arborSize = readBack.getNumDataPatches() * numItemsInPatch;
      for (int d = 0; d < arborSize; d++) {
         w[d] = std::numeric_limits<float>::infinity();
      }
   }

   // Read the data back and verify timestamp.
   auto checkpointReader = std::make_shared<CheckpointEntryWeightPvp>(
         readBack.getName(), &readBack, compressFlag);
   double readTime = (double)(timestamp == 0.0);
   checkpointReader->read(fileManager, &readTime);
   FatalIf(
         readTime != timestamp,
         "Timestamp read from checkpoint was %f; expected %f\n",
         readTime,
         timestamp);

   // Compare the weight data
   for (int a = 0; a < numArbors; a++) {
      float *weightsArborStart = weights.getData(a);
      float *readBackArborStart = readBack.getData(a);
      for (int p = 0; p < numDataPatches; p++) {
         int globalPatchIndex;
         if (sharedFlag) {
            globalPatchIndex = p;
         }
         else {
            globalPatchIndex = calcGlobalPatchIndex(p, mpiBlock, preLoc, postLoc, nxp, nyp);
         }
         Patch const &patch = weights.getPatch(p);
         for (int k = 0; k < numItemsInPatch; k++) {
            if (sharedFlag or isActiveWeight(patch, nxp, nyp, nfp, k)) {
               int const indexIntoArbor = p * numItemsInPatch + k;
               float weightValue = weightsArborStart[indexIntoArbor];
               float readBackValue = readBackArborStart[indexIntoArbor];
               FatalIf(
                     readBackValue != weightValue,
                     "%s, Rank %d, patch %d (global %d), patch item %d: "
                     "expected %f; observed %f (discrepancy %g)\n",
                     label.c_str(),
                     mpiBlock->getGlobalRank(),
                     p,
                     globalPatchIndex,
                     k,
                     (double)weightValue,
                     (double)readBackValue,
                     (double)(readBackValue - weightValue));
            }
         }
      }
   }

   if (mpiBlock->getRank() == 0) {
      InfoLog().printf("%s passed.\n", label.c_str());
   }
}

PVLayerLoc setLayerLoc(
      Communicator const *communicator,
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
      std::shared_ptr<MPIBlock const> mpiBlock,
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
   x += preLoc.nx * (mpiBlock->getStartColumn() + mpiBlock->getColumnIndex());
   int y = kyPos(localPatchIndex, numPatchesX, numPatchesY, nf);
   y += preLoc.ny * (mpiBlock->getStartRow() + mpiBlock->getRowIndex());
   int f = featureIndex(localPatchIndex, numPatchesX, numPatchesY, nf);

   int patchIndexGlobal = kIndex(x, y, f, numPatchesXGlobal, numPatchesYGlobal, nf);
   return patchIndexGlobal;
}

MPIBlock const setMPIBlock(PV_Init &pv_initObj) {
   std::shared_ptr<Arguments const> arguments     = pv_initObj.getArguments();
   auto globalMPIBlock = pv_initObj.getCommunicator()->getGlobalMPIBlock();
   Checkpointer tempCheckpointer("column", pv_initObj.getCommunicator(), arguments);
   MPIBlock const mpiBlock = *tempCheckpointer.getMPIBlock();
   return mpiBlock;
}

float calcWeight(int patchIndex, int itemIndex, int numItemsInPatch) {
   return (float)patchIndex + (float)itemIndex / (float)numItemsInPatch;
}

bool isActiveWeight(Patch const &patch, int nxp, int nyp, int nfp, int itemIndex) {
   int const x = kxPos(itemIndex, nxp, nyp, nfp);
   int const y = kyPos(itemIndex, nxp, nyp, nfp);

   int const startx = kxPos(patch.offset, nxp, nyp, nfp);
   int const starty = kyPos(patch.offset, nxp, nyp, nfp);

   return (x >= startx and x < startx + patch.nx and y >= starty and y < starty + patch.ny);
}
