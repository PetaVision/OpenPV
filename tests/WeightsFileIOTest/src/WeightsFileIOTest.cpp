/*
 * WeightsFileIOTest.cpp
 */

#include <columns/PV_Init.hpp>
#include <io/WeightsFileIO.hpp>
#include <utils/PVLog.hpp>

PV::Weights makeWeights(PV::PV_Init &pv_init, std::string const &name, bool sharedFlag) {
   int const numColumns = pv_init.getCommunicator()->numCommColumns();
   int const numRows    = pv_init.getCommunicator()->numCommRows();

   PVLayerLoc preLoc;
   preLoc.nxGlobal = 8;
   preLoc.nyGlobal = 8;
   preLoc.nx       = preLoc.nxGlobal / numColumns;
   preLoc.ny       = preLoc.nyGlobal / numRows;
   preLoc.nf       = 3;
   preLoc.halo.lt  = 6;
   preLoc.halo.rt  = 6;
   preLoc.halo.dn  = 6;
   preLoc.halo.up  = 6;

   PVLayerLoc postLoc;
   postLoc.nxGlobal = 4;
   postLoc.nyGlobal = 4;
   postLoc.nx       = postLoc.nxGlobal / numColumns;
   postLoc.ny       = postLoc.nyGlobal / numRows;
   postLoc.nf       = 10;
   postLoc.halo.lt  = 0;
   postLoc.halo.rt  = 0;
   postLoc.halo.dn  = 0;
   postLoc.halo.up  = 0;
   // Other fields of preLoc, postLoc are not used.

   int nxp       = 7;
   int nyp       = 7;
   int nfp       = 10;
   int numArbors = 4;

   int xStride = preLoc.nx / postLoc.nx;
   int yStride = preLoc.ny / postLoc.ny;

   PV::Weights weightsObject(name, nxp, nyp, nfp, &preLoc, &postLoc, numArbors, sharedFlag, 0.0);
   weightsObject.allocateDataStructures();
   return weightsObject;
}

int convertLocalIndexToGlobal(int patchIndex, PV::Weights &weights, PV::MPIBlock const *mpiBlock) {
   int numDataPatchesX = weights.getNumDataPatchesX();
   int numDataPatchesY = weights.getNumDataPatchesY();
   int numDataPatchesF = weights.getNumDataPatchesF();
   int xLocal          = kxPos(patchIndex, numDataPatchesX, numDataPatchesY, numDataPatchesF);
   int yLocal          = kyPos(patchIndex, numDataPatchesX, numDataPatchesY, numDataPatchesF);
   int fLocal = featureIndex(patchIndex, numDataPatchesX, numDataPatchesY, numDataPatchesF);

   PVLayerLoc const &preLoc = weights.getGeometry()->getPreLoc();

   int columnIndex = mpiBlock->getStartColumn() + mpiBlock->getColumnIndex();
   int xGlobal     = xLocal + columnIndex * preLoc.nx;

   int rowIndex = mpiBlock->getStartRow() + mpiBlock->getRowIndex();
   int yGlobal  = yLocal + rowIndex * preLoc.ny;

   int fGlobal = fLocal;

   int numGlobalPatchesX =
         preLoc.nx * mpiBlock->getGlobalNumColumns() + preLoc.halo.lt + preLoc.halo.rt;
   int numGlobalPatchesY =
         preLoc.ny * mpiBlock->getGlobalNumRows() + preLoc.halo.dn + preLoc.halo.up;
   int numGlobalPatchesF = preLoc.nf;

   int globalIndex =
         kIndex(xGlobal, yGlobal, fGlobal, numGlobalPatchesX, numGlobalPatchesY, numGlobalPatchesF);
   return globalIndex;
}

bool isActiveWeight(int itemIndex, int localPatchIndex, PV::Weights const &weights) {
   int const nxp = weights.getPatchSizeX();
   int const nyp = weights.getPatchSizeY();
   int const nfp = weights.getPatchSizeF();

   int const x = kxPos(itemIndex, nxp, nyp, nfp);
   int const y = kyPos(itemIndex, nxp, nyp, nfp);

   PV::Patch const &patch = weights.getPatch(localPatchIndex);
   int const startx       = kxPos(patch.offset, nxp, nyp, nfp);
   int const starty       = kyPos(patch.offset, nxp, nyp, nfp);

   return (x >= startx and x < startx + patch.nx and y >= starty and y < starty + patch.ny);
}

void testWeights(PV::Weights &weights, PV::PV_Init &pv_init, bool sharedFlag, bool compressedFlag) {
   int const numArbors         = weights.getNumArbors();
   int const numDataPatches    = weights.getNumDataPatches();
   int const nxp               = weights.getPatchSizeX();
   int const nyp               = weights.getPatchSizeY();
   int const nfp               = weights.getPatchSizeF();
   int const numItemsPerPatch  = nxp * nyp * nfp;
   int const numWeightsInArbor = numDataPatches * numItemsPerPatch;

   double const timestamp = 10.0;

   // Create a checkpointer in order to use the MPIBlock and block-dependent path name
   PV::Checkpointer tempCheckpointer(
         weights.getName(), pv_init.getCommunicator()->getGlobalMPIBlock(), pv_init.getArguments());
   std::string filename         = weights.getName() + std::string(".pvp");
   std::string path             = tempCheckpointer.makeOutputPathFilename(filename);
   PV::MPIBlock const *mpiBlock = tempCheckpointer.getMPIBlock();

   char pathCopy[path.size() + 1];
   std::memcpy(pathCopy, path.c_str(), path.size());
   pathCopy[path.size()] = '\0';
   char *dirName         = dirname(pathCopy);
   ensureDirExists(mpiBlock, dirName);

   // Set weights
   if (sharedFlag) {
      for (int arbor = 0; arbor < numArbors; arbor++) {
         for (int patchIndex = 0; patchIndex < numDataPatches; patchIndex++) {
            float *data         = weights.getDataFromDataIndex(arbor, patchIndex);
            int weightIndexBase = arbor * numWeightsInArbor + patchIndex * numItemsPerPatch;
            for (int w = 0; w < numItemsPerPatch; w++) {
               data[w] = (float)(weightIndexBase + w + 1);
            }
         }
      }
   }
   else {
      for (int arbor = 0; arbor < numArbors; arbor++) {
         for (int localPatchIndex = 0; localPatchIndex < numDataPatches; localPatchIndex++) {
            float *data          = weights.getDataFromDataIndex(arbor, localPatchIndex);
            int globalPatchIndex = convertLocalIndexToGlobal(localPatchIndex, weights, mpiBlock);
            int weightIndexBase  = arbor * numWeightsInArbor + globalPatchIndex * numItemsPerPatch;
            for (int w = 0; w < numItemsPerPatch; w++) {
               if (isActiveWeight(w, localPatchIndex, weights)) {
                  data[w] = (float)(weightIndexBase + w + 1);
               }
               else {
                  data[w] = -1.0f;
               }
            }
         }
      }
   }

   // Write weights
   PV::FileStream *writeStream = nullptr;
   if (mpiBlock->getRank() == 0) {
      writeStream = new PV::FileStream(path.c_str(), std::ios_base::out, false);
   }
   PV::WeightsFileIO weightsFileWrite(writeStream, mpiBlock, &weights);
   weightsFileWrite.writeWeights(timestamp, false);
   delete writeStream;

   // Change weights in memory, to ensure that reading back is doing anything
   for (int arbor = 0; arbor < numArbors; arbor++) {
      float *data = weights.getData(arbor);
      for (int w = 0; w < numWeightsInArbor; w++) {
         data[w] = -1.0f;
      }
   }

   // Read weights back
   PV::FileStream *readStream = nullptr;
   if (mpiBlock->getRank() == 0) {
      readStream = new PV::FileStream(path.c_str(), std::ios_base::in, false);
   }
   PV::WeightsFileIO weightsFileRead(readStream, mpiBlock, &weights);
   int const frameNumber = 0;
   double readTimestamp  = weightsFileRead.readWeights(frameNumber);
   delete readStream;

   // Verify timestamp
   FatalIf(
         readTimestamp != timestamp,
         "%s: timestamp read was %f instead of the expected %f\n",
         weights.getName().c_str(),
         (double)readTimestamp,
         (double)timestamp);

   // Verify weights
   int status = PV_SUCCESS;
   if (sharedFlag) {
      for (int arbor = 0; arbor < numArbors; arbor++) {
         for (int patchIndex = 0; patchIndex < numDataPatches; patchIndex++) {
            float *data         = weights.getDataFromDataIndex(arbor, patchIndex);
            int weightIndexBase = arbor * numWeightsInArbor + patchIndex * numItemsPerPatch;
            for (int w = 0; w < numItemsPerPatch; w++) {
               float correctWeight = (float)(weightIndexBase + w + 1);
               if (data[w] != correctWeight) {
                  status = PV_FAILURE;
                  ErrorLog().printf(
                        "Arbor %d, patch index %d, weight %d: value is %f instead of the expected "
                        "%f\n",
                        arbor,
                        patchIndex,
                        w,
                        (double)data[w],
                        (double)correctWeight);
               }
            }
         }
      }
   }
   else {
      for (int arbor = 0; arbor < numArbors; arbor++) {
         for (int localPatchIndex = 0; localPatchIndex < numDataPatches; localPatchIndex++) {
            float *data          = weights.getDataFromDataIndex(arbor, localPatchIndex);
            int globalPatchIndex = convertLocalIndexToGlobal(localPatchIndex, weights, mpiBlock);
            int weightIndexBase  = arbor * numWeightsInArbor + globalPatchIndex * numItemsPerPatch;
            for (int w = 0; w < numItemsPerPatch; w++) {
               if (isActiveWeight(w, localPatchIndex, weights)) {
                  float correctWeight = (float)(weightIndexBase + w + 1);
                  if (data[w] != correctWeight) {
                     status = PV_FAILURE;
                     ErrorLog().printf(
                           "Arbor %d, patch index %d, weight %d: value is %f instead of the "
                           "expected %f\n",
                           arbor,
                           globalPatchIndex,
                           w,
                           (double)data[w],
                           (double)correctWeight);
                  }
               }
            }
         }
      }
   }
   if (status != PV_SUCCESS) {
      Fatal() << "Test failed for " << weights.getName() << std::endl;
   }
}

void testShared(PV::PV_Init &pv_init) {
   bool const shared         = true;
   bool const noncompressed  = false;
   PV::Weights weightsObject = makeWeights(pv_init, std::string("shared_weights"), shared);
   testWeights(weightsObject, pv_init, shared, noncompressed);
}

void testNonshared(PV::PV_Init &pv_init) {
   bool const nonshared      = false;
   bool const noncompressed  = false;
   PV::Weights weightsObject = makeWeights(pv_init, std::string("nonshared_weights"), nonshared);
   testWeights(weightsObject, pv_init, nonshared, noncompressed);
}

int main(int argc, char *argv[]) {
   PV::PV_Init pv_initObj(&argc, &argv, false /*do not allow unrecognized arguments*/);
   if (pv_initObj.getStringArgument("OutputPath").empty()) {
      pv_initObj.setStringArgument(std::string("OutputPath"), "output");
   }

   testShared(pv_initObj);
   testNonshared(pv_initObj);

   char *programPath = strdup(argv[0]);
   char *programName = basename(programPath);
   InfoLog() << programName << " passed.\n";
   free(programPath);
   return 0;
}
