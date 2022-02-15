/*
 * LocalPatchWeightsFileTest.cpp
 *
 */

#include "ConnectionSpecs.hpp"

#include <columns/PV_Init.hpp>
#include <components/LayerGeometry.hpp> // setLocalLayerLocFields static member function
#include <components/PatchGeometry.hpp> // calcPatchData static member function
#include <include/PVLayerLoc.h>
#include <io/FileManager.hpp>
#include <io/LocalPatchWeightsFile.hpp>
#include <structures/MPIBlock.hpp>
#include <structures/Patch.hpp>
#include <utils/BufferUtilsPvp.hpp>
#include <utils/requiredConvolveMargin.hpp>

#include <algorithm> // used by calcMinVal and calcMaxVal
#include <memory>
#include <limits> // used by calcMinVal and calcMaxVal

using namespace PV;

const float tolerance = 2.5e-7f; // relative error tolerance in comparing weights 

std::shared_ptr<WeightData> allocateWeights(
      int numArbors, int nxp, int nyp, int nfp,
      PVLayerLoc const &preLoc, PVLayerLoc const &postLoc);

float calcMinVal(std::shared_ptr<WeightData const> weightData);
float calcMaxVal(std::shared_ptr<WeightData const> weightData);

// Recursively deletes the contents of the directory specified by path, and removes the directory
// itself, unless path is "." or ends in "/."
int cleanDirectory(std::shared_ptr<FileManager const> fileManager, std::string const &path);

int compareWeights(
      std::shared_ptr<WeightData const> weights1, std::shared_ptr<WeightData const> weights2,
      int nxRestrictedPre, int nyRestrictedPre, int nxRestrictedPost, int nyRestrictedPost,
      std::string const &label);

PVLayerLoc createLayerLoc(
      PV_Init const &pv_Init,
      int nxGlobal, int nyGlobal, int nf, int xMargin, int yMargin, std::string const &label);

std::shared_ptr<WeightData> createWgts1(
      int numArbors, int nxp, int nyp, int nfp,
      PVLayerLoc const &preLoc, PVLayerLoc const &postLoc);
std::shared_ptr<WeightData> createWgts2(
      int numArbors, int nxp, int nyp, int nfp,
      PVLayerLoc const &preLoc, PVLayerLoc const &postLoc);
std::shared_ptr<WeightData> createWgts3(
      int numArbors, int nxp, int nyp, int nfp,
      PVLayerLoc const &preLoc, PVLayerLoc const &postLoc);
std::shared_ptr<WeightData> createWgts4(
      int numArbors, int nxp, int nyp, int nfp,
      PVLayerLoc const &preLoc, PVLayerLoc const &postLoc);

std::shared_ptr<FileManager> createFileManager(PV_Init &pv_init_obj);

std::shared_ptr<WeightData> readFromFileStream(
      std::shared_ptr<FileStream> fileStream,
      int frameNumber,
      std::shared_ptr<FileManager const> fileManager);
void writeToFileStream(
      std::shared_ptr<FileStream> &fileStream,
      std::shared_ptr<WeightData const> weightData,
      PVLayerLoc const &preLayerLoc, PVLayerLoc const &postLayerLoc,
      double timestamp,
      std::shared_ptr<FileManager const> fileManager);

int run(
      std::shared_ptr<FileManager const> fileManager,
      ConnectionSpecs const &connection,
      PV_Init const &pv_init,
      std::string const &directory);

int main(int argc, char *argv[]) {
   int status = PV_SUCCESS;

   PV_Init pv_init(&argc, &argv, false /* do not allow extra arguments */);
   std::shared_ptr<FileManager> fileManager = createFileManager(pv_init);

   // Delete contents of old output directory, to start with a clean slate.
   cleanDirectory(fileManager, std::string("."));

   if (status == PV_SUCCESS) {
      ConnectionSpecs connection(
            2 /*numArbors*/, 5 /*nxp*/, 3 /*nyp*/, 3 /*nfp*/,
            16 /*global restricted nxPre*/, 8 /*global restricted nyPre*/, 4 /*nfPre*/,
            16 /*global restricted nxPost*/, 8 /*global restricted nyPost*/);
      std::string testDesc("one-to-one");
      status = run(fileManager, connection, pv_init, testDesc);
   }
   if (status == PV_SUCCESS) {
      ConnectionSpecs connection(
            2 /*numArbors*/, 5 /*nxp*/, 3 /*nyp*/, 3 /*nfp*/,
            16 /*global restricted nxPre*/, 8 /*global restricted nyPre*/, 4 /*nfPre*/,
            8 /*global restricted nxPost*/, 4 /*global restricted nyPost*/);
      std::string testDesc("many-to-one");
      status = run(fileManager, connection, pv_init, testDesc);
   }
   if (status == PV_SUCCESS) {
      ConnectionSpecs connection(
            2 /*numArbors*/, 6 /*nxp*/, 4 /*nyp*/, 3 /*nfp*/,
            8 /*global restricted nxPre*/, 4 /*global restricted nyPre*/, 4 /*nfPre*/,
            16 /*global restricted nxPost*/, 8 /*global restricted nyPost*/);
      std::string testDesc("one-to-many");
      status = run(fileManager, connection, pv_init, testDesc);
   }

   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int run(
      std::shared_ptr<FileManager const> fileManager,
      ConnectionSpecs const &connection,
      PV_Init const &pv_init,
      std::string const &directory) {
   int status         = PV_SUCCESS;

   int numArbors      = connection.getNumArbors();
   int nxp            = connection.getPatchSizeX();
   int nyp            = connection.getPatchSizeY();
   int nfp            = connection.getPatchSizeF();
   int nxPre          = connection.getNxGlobalRestrictedPre();
   int nyPre          = connection.getNyGlobalRestrictedPre();
   int nfPre          = connection.getNfPre();
   int nxPost         = connection.getNxGlobalRestrictedPost();
   int nyPost         = connection.getNyGlobalRestrictedPost();
   int xMargin        = requiredConvolveMargin(nxPre, nxPost, nxp);
   int yMargin        = requiredConvolveMargin(nyPre, nyPost, nyp);
   PVLayerLoc preLoc  =
        createLayerLoc(pv_init, nxPre, nyPre, nfPre, xMargin, yMargin, std::string("preLoc"));
   PVLayerLoc postLoc =
        createLayerLoc(pv_init, nxPost, nyPost, nfp, 0, 0, std::string("postLoc"));
   double timestamp;

   // Write a shared weights PVP file using the LocalPatchWeightsFile class, and then read it back
   // using primitive FileStream functions, and compare the result.
   fileManager->ensureDirectoryExists(directory); // path is relative to FileManager's baseDir.
   std::string testWritePath = directory + "/testWrite.pvp";

   std::shared_ptr<WeightData> weights1 = createWgts1(numArbors, nxp, nyp, nfp, preLoc, postLoc);

   std::unique_ptr<LocalPatchWeightsFile> wgtFile(new LocalPatchWeightsFile(
      fileManager,
      testWritePath,
      weights1,
      &preLoc,
      &postLoc,
      true /*fileExtendedFlag*/,
      false /*compressedFlag*/,
      false /*readOnlyFlag*/,
      false /*clobberFlag*/,
      false /*verifyWrites*/));

   timestamp = 10.0;
   wgtFile->write(*weights1, timestamp);

   std::shared_ptr<WeightData> weights2 = createWgts2(numArbors, nxp, nyp, nfp, preLoc, postLoc);
   timestamp = 15.0;
   wgtFile->write(*weights2, timestamp);

   wgtFile = std::unique_ptr<LocalPatchWeightsFile>();

   // Now, read the weights back, without using LocalPatchWeightsFile, and compare the results to
   // weights1 and weights2
   auto checkWriteFile = fileManager->open(testWritePath, std::ios_base::in | std::ios_base::binary);
   if (status == PV_SUCCESS) {
      auto checkWeights1 = readFromFileStream(checkWriteFile, 0/*frame number*/, fileManager);
      status = compareWeights(
            weights1, checkWeights1, preLoc.nx, preLoc.ny, postLoc.nx, postLoc.ny,
            std::string("Write test, frame 0"));
   }
   if (status == PV_SUCCESS) {
      auto checkWeights2 = readFromFileStream(checkWriteFile, 1/*frame number*/, fileManager);
      status = compareWeights(
            weights2, checkWeights2, preLoc.nx, preLoc.ny, postLoc.nx, postLoc.ny,
            std::string("Write test, frame 1"));
   }
   if (status != PV_SUCCESS) { return EXIT_FAILURE; }

   // Write a shared weights PVP file using primitive FileStream functions, and then read it back
   // using the LocalPatchWeightsFile class, and compare the result.
   //
   auto weights3 = createWgts3(numArbors, nxp, nyp, nfp, preLoc, postLoc);
   auto weights4 = createWgts4(numArbors, nxp, nyp, nfp, preLoc, postLoc);
   std::string testReadPath = directory + "/testRead.pvp";
   // File shouldn't exist; create it.
   auto testReadFile = fileManager->open(testReadPath, std::ios_base::out);
   auto mode = std::ios_base::in | std::ios_base::out | std::ios_base::binary;
   testReadFile = fileManager->open(testReadPath, mode); // closes & reopens with read/write mode
   double timestamp3 = 20.0;
   writeToFileStream(testReadFile, weights3, preLoc, postLoc, timestamp3, fileManager);
   double timestamp4 = 25.0;
   writeToFileStream(testReadFile, weights4, preLoc, postLoc, timestamp4, fileManager);
   testReadFile = nullptr; // closes file

   // Now read the weights using the LocalPatchWeightsFile class, and compare the results
   wgtFile = std::unique_ptr<LocalPatchWeightsFile>(new LocalPatchWeightsFile(
      fileManager,
      testReadPath,
      weights3,
      &preLoc,
      &postLoc,
      true /*fileExtendedFlag*/,
      false /*compressedFlag*/,
      true /*readOnlyFlag*/,
      false /*clobberFlag*/,
      false /*verifyWrites*/));
   double readTimestamp3;
   if (status == PV_SUCCESS) {
      auto readWeights3 = std::make_shared<WeightData>(
         numArbors, nxp, nyp, nfp, preLoc.nx + 2*xMargin, preLoc.ny + 2*yMargin, nfPre);
      wgtFile->read(*readWeights3, readTimestamp3);
      status = compareWeights(
            weights3, readWeights3, preLoc.nx, preLoc.ny, postLoc.nx, postLoc.ny,
            std::string("Read test, frame 0"));
   }
   if (status == PV_SUCCESS) {
      if (readTimestamp3 != timestamp3) {
         ErrorLog().printf("Read test, frame 0, expected timestamp %f, received %f\n",
               timestamp3, readTimestamp3);
         status = PV_FAILURE;
      }
   }
   double readTimestamp4;
   if (status == PV_SUCCESS) {
      auto readWeights4 = std::make_shared<WeightData>(
         numArbors, nxp, nyp, nfp, nxPre + 2*xMargin, nyPre + 2*yMargin, nfPre);
      wgtFile->read(*readWeights4, readTimestamp4);
      status = compareWeights(
            weights4, readWeights4, preLoc.nx, preLoc.ny, postLoc.nx, postLoc.ny,
            std::string("Read test, frame 1"));
   }
   if (status == PV_SUCCESS) {
      if (readTimestamp4 != timestamp4) {
         ErrorLog().printf("Read test, frame 1, expected timestamp %f, received %f\n",
               timestamp4, readTimestamp4);
         status = PV_FAILURE;
      }
   }

   if (status == PV_SUCCESS) {
      InfoLog() << "Test passed.\n";
   }
   else {
      Fatal() << "Test failed.\n";
   }
   return status;
}

std::shared_ptr<WeightData> allocateWeights(
      int numArbors, int nxp, int nyp, int nfp,
      PVLayerLoc const &preLoc, PVLayerLoc const &postLoc) {
   int nxPreRestricted = preLoc.nx;
   int nxPost          = postLoc.nx;
   int xMargin         = requiredConvolveMargin(nxPreRestricted, nxPost, nxp);
   int nxPreExtended   = nxPreRestricted + 2 * xMargin;
   int nyPreRestricted = preLoc.ny;
   int nyPost          = postLoc.ny;
   int yMargin         = requiredConvolveMargin(nyPreRestricted, nyPost, nyp);
   int nyPreExtended   = nyPreRestricted + 2 * yMargin;
   auto weightData     = std::make_shared<WeightData>(
         numArbors, nxp, nyp, nfp, nxPreExtended, nyPreExtended, preLoc.nf);
   return weightData;
}

float calcMinVal(std::shared_ptr<WeightData const> weightData) {
   float minVal = std::numeric_limits<float>::infinity();
   long numElements = weightData->getPatchSizeOverall() * weightData->getNumDataPatchesOverall();
   for (int a = 0; a < weightData->getNumArbors(); ++a) {
      float const *firstElement = weightData->getData(a);
      float const *lastElement = &firstElement[numElements];
      auto minLoc = std::min_element(firstElement, lastElement);
      if (minLoc != lastElement) {
         float arborMin = *minLoc;
         minVal = std::min(arborMin, minVal);
      }
   }
   return minVal;
}

float calcMaxVal(std::shared_ptr<WeightData const> weightData) {
   float maxVal = -std::numeric_limits<float>::infinity();
   long numElements = weightData->getPatchSizeOverall() * weightData->getNumDataPatchesOverall();
   for (int a = 0; a < weightData->getNumArbors(); ++a) {
      float const *firstElement = weightData->getData(a);
      float const *lastElement = &firstElement[numElements];
      auto maxLoc = std::max_element(firstElement, lastElement);
      if (maxLoc != lastElement) {
         float arborMax = *maxLoc;
         maxVal = std::max(arborMax, maxVal);
      }
   }
   return maxVal;
}

int cleanDirectory(std::shared_ptr<FileManager const> fileManager, std::string const &path) {
   int status = PV_SUCCESS;
   if (fileManager->isRoot()) {
      struct stat statbuf;
      status = fileManager->stat(path, statbuf);
      if (status and errno == ENOENT) { return PV_SUCCESS; }
      auto dirContents = fileManager->listDirectory(path);
      for (auto &d : dirContents) {
         std::string dirEntry(path + "/" + d);
         int status = fileManager->stat(dirEntry, statbuf);
         if (status) {
            ErrorLog().printf("Unable to stat \"%s\": %s\n", dirEntry.c_str(), strerror(errno));
            status = PV_FAILURE;
            break;
         }
         if (statbuf.st_mode & S_IFREG) {
            fileManager->deleteFile(dirEntry);
         }
         else if (statbuf.st_mode & S_IFDIR) {
            status = cleanDirectory(fileManager, dirEntry);
            if (status != PV_SUCCESS) { break; }
         }
      }
      std::string dotAtEnd("/.");
      if (path != ".") {
         if (path.size() < dotAtEnd.size() or
             path.substr(path.size() - dotAtEnd.size()) != dotAtEnd) {
            fileManager->deleteDirectory(path);
         }
      }
   }
   return status;
}

int compareWeights(
      std::shared_ptr<WeightData const> weights1, std::shared_ptr<WeightData const> weights2,
      int nxRestrictedPre, int nyRestrictedPre, int nxRestrictedPost, int nyRestrictedPost,
      std::string const &label) {
   int status = PV_SUCCESS;
   if (weights1->getNumArbors() != weights2->getNumArbors()) {
      ErrorLog().printf(
            "compareWeights: numbers of arbors differ (%d versus %d)\n",
            weights1->getNumArbors(), weights2->getNumArbors());
      status = PV_FAILURE;
   }
   int numArbors = weights1->getNumArbors();

   if (weights1->getPatchSizeX() != weights2->getPatchSizeX()) {
      ErrorLog().printf(
            "compareWeights: PatchSizeX differs (%d versus %d)\n",
            weights1->getPatchSizeX(), weights2->getPatchSizeX());
      status = PV_FAILURE;
   }
   int patchSizeX = weights1->getPatchSizeX();
   if (weights1->getPatchSizeY() != weights2->getPatchSizeY()) {
      ErrorLog().printf(
            "compareWeights: PatchSizeY differs (%d versus %d)\n",
            weights1->getPatchSizeY(), weights2->getPatchSizeY());
      status = PV_FAILURE;
   }
   int patchSizeY = weights1->getPatchSizeY();
   if (weights1->getPatchSizeF() != weights2->getPatchSizeF()) {
      ErrorLog().printf(
            "compareWeights: PatchSizeF differs (%d versus %d)\n",
            weights1->getPatchSizeF(), weights2->getPatchSizeF());
      status = PV_FAILURE;
   }
   int patchSizeF = weights1->getPatchSizeF();

   int xMargin = requiredConvolveMargin(nxRestrictedPre, nxRestrictedPost, patchSizeX);
   int yMargin = requiredConvolveMargin(nyRestrictedPre, nyRestrictedPost, patchSizeY);
   if (weights1->getNumDataPatchesX() < nxRestrictedPre + 2 * xMargin) {
      ErrorLog().printf(
            "compareWeights, %s: weights1 does not have enough patches in the x-direction "
            "(nxRestricted = %d, required margins %d, but weights1 is only %d patches wide)\n",
            label.c_str(), nxRestrictedPre, xMargin, weights1->getNumDataPatchesX());
      status = PV_FAILURE;
   }
   if (weights1->getNumDataPatchesY() < nyRestrictedPre + 2 * yMargin) {
      ErrorLog().printf(
            "compareWeights, %s: weights1 does not have enough patches in the y-direction "
            "(nyRestricted = %d, required margins %d, but weights1 is only %d patches high)\n",
            label.c_str(), nyRestrictedPre, yMargin, weights1->getNumDataPatchesY());
      status = PV_FAILURE;
   }
   if (weights2->getNumDataPatchesX() < nxRestrictedPre + 2 * xMargin) {
      ErrorLog().printf(
            "compareWeights, %s: weights2 does not have enough patches in the x-direction "
            "(nxRestricted = %d, required margins %d, but weights2 is only %d patches wide)\n",
            label.c_str(), nxRestrictedPre, xMargin, weights2->getNumDataPatchesX());
      status = PV_FAILURE;
   }
   if (weights2->getNumDataPatchesY() < nyRestrictedPre + 2 * yMargin) {
      ErrorLog().printf(
            "compareWeights, %s: weights2 does not have enough patches in the y-direction "
            "(nyRestricted = %d, required margins %d, but weights2 is only %d patches high)\n",
            label.c_str(), nyRestrictedPre, yMargin, weights2->getNumDataPatchesY());
      status = PV_FAILURE;
   }
   if (weights1->getNumDataPatchesF() != weights2->getNumDataPatchesF()) {
      ErrorLog().printf(
            "compareWeights: NumDataPatchesF differs (%d versus %d)\n",
            weights1->getNumDataPatchesF(), weights2->getNumDataPatchesF());
      status = PV_FAILURE;
   }
   int nf = weights1->getNumDataPatchesF();

   int xStartIndex1 = (weights1->getNumDataPatchesX() - nxRestrictedPre) / 2;
   int yStartIndex1 = (weights1->getNumDataPatchesY() - nyRestrictedPre) / 2;
   int xStartIndex2 = (weights2->getNumDataPatchesX() - nxRestrictedPre) / 2;
   int yStartIndex2 = (weights2->getNumDataPatchesY() - nyRestrictedPre) / 2;

   for (int a = 0; a < numArbors; ++a) {
      for (int y = 0; y < nyRestrictedPre + 2 * yMargin; ++y) {
         for (int x = 0; x < nxRestrictedPre + 2 * xMargin; ++x) {
            for (int f = 0; f < nf; ++f) {
               float const *patch1 =
                     weights1->getDataFromXYF(a, x + xStartIndex1, y + yStartIndex1, f);
               float const *patch2 =
                     weights2->getDataFromXYF(a, x + xStartIndex2, y + yStartIndex2, f);
               // Need to compute valid region; should be same for weights 1 and 2
               int xPatchDim, xPatchStart, yPatchDim, yPatchStart, dummy1, dummy2, dummy3;
               PatchGeometry::calcPatchData(
                     x + xStartIndex1,
                     nxRestrictedPre,
                     xStartIndex1,
                     xStartIndex1,
                     nxRestrictedPost,
                     0, 0,
                     patchSizeX,
                     &xPatchDim, &xPatchStart, &dummy1, &dummy2, &dummy3);
               PatchGeometry::calcPatchData(
                     y + yStartIndex1,
                     nyRestrictedPre,
                     yStartIndex1,
                     yStartIndex1,
                     nyRestrictedPost,
                     0, 0,
                     patchSizeY,
                     &yPatchDim, &yPatchStart, &dummy1, &dummy2, &dummy3);
               for (int ky = yPatchStart; ky < yPatchStart + yPatchDim; ++ky) {
                  for (int kx = xPatchStart; kx < xPatchStart + xPatchDim; ++kx) {
                     for (int kf = 0; kf < patchSizeF; ++kf) {
                        int index = kIndex(kx, ky, kf, patchSizeX, patchSizeY, patchSizeF);
                        float discrepancy = patch2[index] - patch1[index];
                        if (std::abs(discrepancy) > tolerance * std::abs(patch1[index])) {
                           ErrorLog().printf(
                                 "compareWeights, %s: weights do not agree at patch with "
                                 "arbor %d, restricted index x=%d, y=%d, f=%d, patch element at "
                                 "x=%d, y=%d, f=%d (%f versus %f, discrepancy %g)\n",
                                 label.c_str(), a, x, y, f, kx, ky, kf,
                                 (double)patch1[index], (double)patch2[index],
                                 (double)discrepancy);
                           status = PV_FAILURE;
                        }
                     }
                  }
               }
            }
         }
      }
   }

   return status;
}

PVLayerLoc createLayerLoc(
      PV_Init const &pv_Init,
      int nxGlobal, int nyGlobal, int nf, int xMargin, int yMargin, std::string const &label) {
   PVLayerLoc loc;
   loc.nbatchGlobal = pv_Init.getCommunicator()->numCommBatches();
   loc.nxGlobal     = nxGlobal;
   loc.nyGlobal     = nyGlobal;
   loc.nf           = nf;
   LayerGeometry::setLocalLayerLocFields(&loc, pv_Init.getCommunicator(), label);
   loc.halo.lt = xMargin;
   loc.halo.rt = xMargin;
   loc.halo.dn = yMargin;
   loc.halo.up = yMargin;
   return loc;
}

std::shared_ptr<WeightData> createWgts1(
      int numArbors, int nxp, int nyp, int nfp,
      PVLayerLoc const &preLoc, PVLayerLoc const &postLoc) {
   auto weightData       = allocateWeights(numArbors, nxp, nyp, nfp, preLoc, postLoc);
   int nxLocalExt        = preLoc.nx + preLoc.halo.lt + preLoc.halo.rt;
   int nyLocalExt        = preLoc.ny + preLoc.halo.dn + preLoc.halo.up;
   int nf                = preLoc.nf;
   int nxGlobalExt       = preLoc.nxGlobal + preLoc.halo.lt + preLoc.halo.rt;
   int nyGlobalExt       = preLoc.nyGlobal + preLoc.halo.dn + preLoc.halo.up;
   long numPatchesGlobal = static_cast<long>(nxGlobalExt * nyGlobalExt * nf);
   long numPatchesLocal  = static_cast<long>(nxLocalExt * nyLocalExt * nf); 
   long patchSizeOverall = static_cast<long>(weightData->getPatchSizeOverall());
   long numValuesOverall = patchSizeOverall * numPatchesGlobal * numArbors;
   for (int a = 0; a < numArbors; ++a) {
      for (long k = 0; k < numPatchesLocal; ++k) {
         int xLocal            = kxPos(k, nxLocalExt, nyLocalExt, nf);
         int yLocal            = kyPos(k, nxLocalExt, nyLocalExt, nf);
         int fIndex            = featureIndex(k, nxLocalExt, nyLocalExt, nf);
         int xGlobal           = xLocal + preLoc.kx0;
         int yGlobal           = yLocal + preLoc.ky0;
         int kGlobal           = kIndex(xGlobal, yGlobal, fIndex, nxGlobalExt, nyGlobalExt, nf);
         long baseIndexGlobal  = patchSizeOverall * (a * numPatchesGlobal + kGlobal);
         float *patchLocation  = weightData->getDataFromDataIndex(a, k);
         for (long p = 0; p < patchSizeOverall; ++p) {
            float indexGlobal = static_cast<float>(baseIndexGlobal + p + 1); 
            patchLocation[p] = indexGlobal;
         }
      }
   }
   return weightData;
}

std::shared_ptr<WeightData> createWgts2(
      int numArbors, int nxp, int nyp, int nfp,
      PVLayerLoc const &preLoc, PVLayerLoc const &postLoc) {
   auto weightData       = allocateWeights(numArbors, nxp, nyp, nfp, preLoc, postLoc);
   int nxLocalExt        = preLoc.nx + preLoc.halo.lt + preLoc.halo.rt;
   int nyLocalExt        = preLoc.ny + preLoc.halo.dn + preLoc.halo.up;
   int nf                = preLoc.nf;
   int nxGlobalExt       = preLoc.nxGlobal + preLoc.halo.lt + preLoc.halo.rt;
   int nyGlobalExt       = preLoc.nyGlobal + preLoc.halo.dn + preLoc.halo.up;
   long numPatchesGlobal = static_cast<long>(nxGlobalExt * nyGlobalExt * nf);
   long numPatchesLocal  = static_cast<long>(nxLocalExt * nyLocalExt * nf); 
   long patchSizeOverall = static_cast<long>(weightData->getPatchSizeOverall());
   long numValuesOverall = patchSizeOverall * numPatchesGlobal * numArbors;
   for (int a = 0; a < numArbors; ++a) {
      for (long k = 0; k < numPatchesLocal; ++k) {
         int xLocal            = kxPos(k, nxLocalExt, nyLocalExt, nf);
         int yLocal            = kyPos(k, nxLocalExt, nyLocalExt, nf);
         int fIndex            = featureIndex(k, nxLocalExt, nyLocalExt, nf);
         int xGlobal           = xLocal + preLoc.kx0;
         int yGlobal           = yLocal + preLoc.ky0;
         int kGlobal           = kIndex(xGlobal, yGlobal, fIndex, nxGlobalExt, nyGlobalExt, nf);
         long baseIndexGlobal  = patchSizeOverall * (a * numPatchesGlobal + kGlobal);
         float *patchLocation  = weightData->getDataFromDataIndex(a, k);
         for (long p = 0; p < patchSizeOverall; ++p) {
            float indexGlobal = static_cast<float>(baseIndexGlobal + p + 1); 
            patchLocation[p] = std::sqrt(indexGlobal);
         }
      }
   }
   return weightData;
}

std::shared_ptr<WeightData> createWgts3(
      int numArbors, int nxp, int nyp, int nfp,
      PVLayerLoc const &preLoc, PVLayerLoc const &postLoc) {
   auto weightData       = allocateWeights(numArbors, nxp, nyp, nfp, preLoc, postLoc);
   int nxLocalExt        = preLoc.nx + preLoc.halo.lt + preLoc.halo.rt;
   int nyLocalExt        = preLoc.ny + preLoc.halo.dn + preLoc.halo.up;
   int nf                = preLoc.nf;
   int nxGlobalExt       = preLoc.nxGlobal + preLoc.halo.lt + preLoc.halo.rt;
   int nyGlobalExt       = preLoc.nyGlobal + preLoc.halo.dn + preLoc.halo.up;
   long numPatchesGlobal = static_cast<long>(nxGlobalExt * nyGlobalExt * nf);
   long numPatchesLocal  = static_cast<long>(nxLocalExt * nyLocalExt * nf); 
   long patchSizeOverall = static_cast<long>(weightData->getPatchSizeOverall());
   long numValuesOverall = patchSizeOverall * numPatchesGlobal * numArbors;
   for (int a = 0; a < numArbors; ++a) {
      for (long k = 0; k < numPatchesLocal; ++k) {
         int xLocal            = kxPos(k, nxLocalExt, nyLocalExt, nf);
         int yLocal            = kyPos(k, nxLocalExt, nyLocalExt, nf);
         int fIndex            = featureIndex(k, nxLocalExt, nyLocalExt, nf);
         int xGlobal           = xLocal + preLoc.kx0;
         int yGlobal           = yLocal + preLoc.ky0;
         int kGlobal           = kIndex(xGlobal, yGlobal, fIndex, nxGlobalExt, nyGlobalExt, nf);
         long baseIndexGlobal  = patchSizeOverall * (a * numPatchesGlobal + kGlobal);
         float *patchLocation  = weightData->getDataFromDataIndex(a, k);
         for (long p = 0; p < patchSizeOverall; ++p) {
            float indexGlobal = static_cast<float>(baseIndexGlobal + p + 1); 
            patchLocation[p] = 1.0f - (indexGlobal - 1) / static_cast<float>(numValuesOverall);
         }
      }
   }
   return weightData;
}

std::shared_ptr<WeightData> createWgts4(
      int numArbors, int nxp, int nyp, int nfp,
      PVLayerLoc const &preLoc, PVLayerLoc const &postLoc) {
   auto weightData       = allocateWeights(numArbors, nxp, nyp, nfp, preLoc, postLoc);
   int nxLocalExt        = preLoc.nx + preLoc.halo.lt + preLoc.halo.rt;
   int nyLocalExt        = preLoc.ny + preLoc.halo.dn + preLoc.halo.up;
   int nf                = preLoc.nf;
   int nxGlobalExt       = preLoc.nxGlobal + preLoc.halo.lt + preLoc.halo.rt;
   int nyGlobalExt       = preLoc.nyGlobal + preLoc.halo.dn + preLoc.halo.up;
   long numPatchesGlobal = static_cast<long>(nxGlobalExt * nyGlobalExt * nf);
   long numPatchesLocal  = static_cast<long>(nxLocalExt * nyLocalExt * nf); 
   long patchSizeOverall = static_cast<long>(weightData->getPatchSizeOverall());
   long numValuesOverall = patchSizeOverall * numPatchesGlobal * numArbors;
   for (int a = 0; a < numArbors; ++a) {
      for (long k = 0; k < numPatchesLocal; ++k) {
         int xLocal            = kxPos(k, nxLocalExt, nyLocalExt, nf);
         int yLocal            = kyPos(k, nxLocalExt, nyLocalExt, nf);
         int fIndex            = featureIndex(k, nxLocalExt, nyLocalExt, nf);
         int xGlobal           = xLocal + preLoc.kx0;
         int yGlobal           = yLocal + preLoc.ky0;
         int kGlobal           = kIndex(xGlobal, yGlobal, fIndex, nxGlobalExt, nyGlobalExt, nf);
         long baseIndexGlobal  = patchSizeOverall * (a * numPatchesGlobal + kGlobal);
         float *patchLocation  = weightData->getDataFromDataIndex(a, k);
         for (long p = 0; p < patchSizeOverall; ++p) {
            float indexGlobal = static_cast<float>(baseIndexGlobal + p + 1); 
            patchLocation[p] = -indexGlobal / static_cast<float>(numValuesOverall);
         }
      }
   }
   return weightData;
}

std::shared_ptr<FileManager> createFileManager(PV_Init &pv_init_obj) {
   auto mpiBlock  = pv_init_obj.getCommunicator()->getIOMPIBlock();
   auto arguments = pv_init_obj.getArguments();
   std::string baseDirectory = arguments->getStringArgument("OutputPath");
   FatalIf(baseDirectory.substr(0, 7) != "output/","OutputPath must begin with \"output\"\n");

   auto fileManager = std::make_shared<FileManager> (mpiBlock, baseDirectory);
   return fileManager;
}

std::shared_ptr<WeightData> readFromFileStream(
      std::shared_ptr<FileStream> fileStream,
      int frameNumber,
      std::shared_ptr<FileManager const> fileManager) {
   auto const &mpiBlock = fileManager->getMPIBlock();
   BufferUtils::WeightHeader header;
   int rootProc = fileManager->getRootProcessRank();
   if (fileManager->isRoot()) {
      fileStream->setInPos(0L, std::ios_base::beg);
      fileStream->read(&header, static_cast<long>(sizeof(header)));

      for (int f = 0; f < frameNumber; ++f) {
         int patchSize = 8L + header.nxp * header.nyp * header.nfp * header.baseHeader.dataSize;
         long frameDataSize =
               static_cast<long>(patchSize * header.numPatches * header.baseHeader.numRecords);
         fileStream->setInPos(frameDataSize, std::ios_base::cur);
         fileStream->read(&header, static_cast<long>(sizeof(header)));
      }
   }
   MPI_Bcast(&header, sizeof(header), MPI_BYTE, rootProc, mpiBlock->getComm());
   int numArbors  = header.baseHeader.numRecords;
   int blockNxExt = header.baseHeader.nxExtended;
   int blockNyExt = header.baseHeader.nyExtended;
   int nfPre      = header.baseHeader.nf;
   int patchSize  = header.nxp * header.nyp * header.nfp;
   pvAssert(header.baseHeader.dataSize == static_cast<int>(sizeof(float))); // TODO: compressed
   long patchSizeBytes  = static_cast<long>(patchSize * header.baseHeader.dataSize);
   long numPatches      = static_cast<long>(blockNxExt * blockNyExt * nfPre);
   auto blockWeightData = std::make_shared<WeightData>(
         numArbors, header.nxp, header.nyp, header.nfp, blockNxExt, blockNyExt, nfPre);

   int marginLeft       = (blockNxExt - header.baseHeader.nx) / 2;
   int marginRight      = blockNxExt - header.baseHeader.nx - marginLeft;
   int marginDown       = (blockNyExt - header.baseHeader.ny) / 2;
   int marginUp         = blockNyExt - header.baseHeader.ny - marginDown;
   int localNx          = header.baseHeader.nx / mpiBlock->getNumColumns();
   int localNy          = header.baseHeader.ny / mpiBlock->getNumRows();
   int localNxExt       = localNx + marginLeft + marginRight;
   int localNyExt       = localNy + marginDown + marginUp;
   auto localWeightData = std::make_shared<WeightData>(
         numArbors, header.nxp, header.nyp, header.nfp, localNxExt, localNyExt, nfPre);
   for (int a = 0; a < numArbors; ++a) {
      if (fileManager->isRoot()) {
         for (long p = 0; p < numPatches; ++p) {
            float *patchAddress = blockWeightData->getDataFromDataIndex(a, p);
            fileStream->setInPos(8L, std::ios_base::cur);
            fileStream->read(patchAddress, patchSizeBytes);
         }
      }

      // Root process broadcasts the entire block's weights.
      // Inefficient, but easier to code, and this part is used only in the test.
      long blockArborSizeBytes = numPatches * patchSizeBytes;
      float *blockArborPointer = blockWeightData->getData(a);
      MPI_Bcast(blockArborPointer, blockArborSizeBytes, MPI_BYTE, rootProc, mpiBlock->getComm());

      // Each process extracts its part of the weights from the entire block
      int xStart    = localNx * mpiBlock->getColumnIndex();
      int yStart    = localNy * mpiBlock->getRowIndex();
      int lineSize  = localNxExt * nfPre * patchSize;
      for (int ky = 0; ky < localNyExt; ++ky) {
         int blockStartIndex = kIndex(xStart, yStart + ky, 0, blockNxExt, blockNyExt, nfPre);
         float *blockLinePointer = blockWeightData->getDataFromDataIndex(a, blockStartIndex);
         int localStartIndex = ky * localNxExt * nfPre;
         float *localLinePointer = localWeightData->getDataFromDataIndex(a, localStartIndex);
         for (int k = 0; k < lineSize; ++k) {
            localLinePointer[k] = blockLinePointer[k];
         }
      }
   }
   return localWeightData;
}

void writeToFileStream(
      std::shared_ptr<FileStream> &fileStream,
      std::shared_ptr<WeightData const> weightData,
      PVLayerLoc const &preLayerLoc, PVLayerLoc const &postLayerLoc,
      double timestamp,
      std::shared_ptr<FileManager const> fileManager) {
   int numArbors = weightData->getNumArbors();
   FatalIf(numArbors == 0, "writeToFileStream() called with empty weights\n");
   int nxp         = weightData->getPatchSizeX();
   int nyp         = weightData->getPatchSizeY();
   int nfp         = weightData->getPatchSizeF();
   int numPatchesX = weightData->getNumDataPatchesX();
   int numPatchesY = weightData->getNumDataPatchesY();
   int numPatchesF = weightData->getNumDataPatchesF();
   float minVal    = calcMinVal(weightData);
   float maxVal    = calcMaxVal(weightData);
   auto mpiBlock   = fileManager->getMPIBlock();
   int nxExtendedLocal   = preLayerLoc.nx + preLayerLoc.halo.lt + preLayerLoc.halo.rt;
   int nyExtendedLocal   = preLayerLoc.ny + preLayerLoc.halo.dn + preLayerLoc.halo.up;
   int numPatchesPerLine = preLayerLoc.nf * nxExtendedLocal;
   long patchSizeOverall = static_cast<long>(weightData->getPatchSizeOverall());
   long lineSize         = static_cast<long>(numPatchesPerLine) * patchSizeOverall;
   long bufferSize       = lineSize * static_cast<long>(nyExtendedLocal);
   if (fileManager->isRoot()) {
      auto weightHeader = BufferUtils::buildNonsharedWeightHeader(
            nxp, nyp, nfp,
            numArbors,
            true /*fileExtendedFlag*/,
            timestamp,
            &preLayerLoc, &postLayerLoc,
            mpiBlock->getNumColumns(), mpiBlock->getNumRows(),
            minVal,
            maxVal,
            false /*compressFlag*/);
      long const headerSize = 104L;
      FatalIf(
            static_cast<long>(sizeof(weightHeader)) != headerSize,
            "Weight header size should be 104 but is %zu\n",
            sizeof(weightHeader));
      fileStream->write(&weightHeader, headerSize);
      long dataStartInFile = fileStream->getOutPos();

      Patch patchHeader;
      patchHeader.nx       = static_cast<uint16_t>(nxp);
      patchHeader.ny       = static_cast<uint16_t>(nyp);
      patchHeader.offset   = static_cast<uint32_t>(0);
      long patchHeaderSize = static_cast<long>(sizeof(patchHeader));
      pvAssert(patchHeaderSize == 8L);
      std::vector<float> mpiBuffer(bufferSize);
      for (int mpiRow = 0; mpiRow < mpiBlock->getNumRows(); ++mpiRow) {
         for (int mpiColumn = 0; mpiColumn < mpiBlock->getNumColumns(); ++mpiColumn) {
            int rank = mpiBlock->calcRankFromRowColBatch(mpiRow, mpiColumn, 0);
            for (int a = 0; a < numArbors; ++a) {
               int nxGlobalExt          = weightHeader.baseHeader.nxExtended;
               int nyGlobalExt          = weightHeader.baseHeader.nyExtended;
               int nf                   = weightHeader.baseHeader.nf;
               long numPatchesGlobalExt = static_cast<long>(nxGlobalExt * nyGlobalExt * nf);
               long dataSize            = static_cast<long>(sizeof(float)); // assumes uncompressed
               long patchSizeBytes      = patchSizeOverall * dataSize;
               long patchSizeInFile     = patchSizeBytes + patchHeaderSize;
               long arborSizeInFile     = patchSizeInFile * numPatchesGlobalExt;
               long arborStartInFile    = dataStartInFile + static_cast<long>(a) * arborSizeInFile;
               if (rank == mpiBlock->getRank()) {
                  size_t numBytes = sizeof(float) * static_cast<size_t>(bufferSize);
                  memcpy(mpiBuffer.data(), weightData->getData(a), numBytes);
               }
               else {
                     MPI_Recv(
                           mpiBuffer.data(), static_cast<int>(bufferSize), MPI_FLOAT,
                           rank, 140 + a /*tag*/, mpiBlock->getComm(), MPI_STATUS_IGNORE);
               }
               int nxLocalExtended = preLayerLoc.nx + preLayerLoc.halo.lt + preLayerLoc.halo.rt;
               int nyLocalExtended = preLayerLoc.ny + preLayerLoc.halo.dn + preLayerLoc.halo.up;
               for (int y = 0; y < nyLocalExtended; ++y) {
                  int lineStartIndexInBlock = kIndex(
                        mpiColumn * preLayerLoc.nx,
                        mpiRow * preLayerLoc.ny + y,
                        0,
                        weightHeader.baseHeader.nxExtended,
                        weightHeader.baseHeader.nyExtended,
                        weightHeader.baseHeader.nf);
                  long lineStartFileOffset =
                        static_cast<long>(patchSizeInFile * lineStartIndexInBlock);
                  long lineStartFilePos = arborStartInFile + lineStartFileOffset;
                  fileStream->setOutPos(lineStartFilePos, std::ios_base::beg);
                  for (int x = 0; x < nxLocalExtended; ++x) {
                     for (int f = 0; f < nf; ++f) {
                        fileStream->write(&patchHeader, patchHeaderSize);
                        int patchIndex = kIndex(x, y, f, nxLocalExtended, nyLocalExtended, nf);
                        long dataIndex = static_cast<long>(patchIndex) * patchSizeOverall;
                        float *patchData = &mpiBuffer.at(dataIndex);
                        fileStream->write(patchData, patchSizeBytes);
                     } // f
                  } // x
               } // y
            } // a
         } // mpiColumn
      } // mpiRow
   } // if isRoot()
   else {
      for (int a = 0; a < numArbors; ++a) {
         MPI_Send(weightData->getData(a), static_cast<int>(bufferSize), MPI_FLOAT,
         0 /*receiving rank*/, 140 + a /*tag*/, mpiBlock->getComm());
      }  
   }
}
