/*
 * LocalPatchWeightsFileTest.cpp
 *
 */

#include "ConnectionSpecs.hpp"

#include <columns/PV_Init.hpp>
#include <components/LayerGeometry.hpp> // setLocalLayerLocFields static member function
#include <structures/PatchGeometry.hpp> // calcPatchData static member function
#include <structures/PVLayerLoc.hpp>
#include <io/FileManager.hpp>
#include <io/LocalPatchWeightsFile.hpp>
#include <structures/MPIBlock.hpp>
#include <structures/Patch.hpp>
#include <utils/BufferUtilsPvp.hpp>
#include <utils/requiredConvolveMargin.hpp>

#include <algorithm> // used by calcMinVal and calcMaxVal
#include <limits> // used by calcMinVal and calcMaxVal
#include <memory>
#include <string>

using namespace PV;

const float tolerance = 2.5e-7f; // relative error tolerance in comparing weights 

std::shared_ptr<WeightData> allocateWeights(
      std::string const &name,
      int numArbors, int nxp, int nyp, int nfp,
      PVLayerLoc const &preLoc, PVLayerLoc const &postLoc);

float calcMinVal(std::shared_ptr<WeightData const> weightData);
float calcMaxVal(std::shared_ptr<WeightData const> weightData);

// Recursively deletes the contents of the directory specified by path, and removes the directory
// itself, unless path is "." or ends in "/."
int cleanDirectory(std::shared_ptr<FileManager const> fileManager, std::string const &path);

int compareWeights(
      std::shared_ptr<WeightData const> expectedWeights,
      std::shared_ptr<WeightData const> observedWeights,
      int nxRestrictedPre, int nyRestrictedPre, int nxRestrictedPost, int nyRestrictedPost,
      std::string const &label);

PVLayerLoc createLayerLoc(
      PV_Init const &pv_Init,
      int nxGlobal, int nyGlobal, int nf, int xMargin, int yMargin, std::string const &label);

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

void setWeights1(
      std::shared_ptr<WeightData> weightData,
      PVLayerLoc const &preLoc,
      PVLayerLoc const &postLoc);

void setWeights2(
      std::shared_ptr<WeightData> weightData,
      PVLayerLoc const &preLoc,
      PVLayerLoc const &postLoc);

int main(int argc, char *argv[]) {
   FatalIf(
         tolerance < 0.0f,
         "LocalPatchWeightsFileTest has tolerance set to %g, but tolerance must be positive.\n",
         (double)tolerance);
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
#ifdef PHALACROCORAX
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
#endif // PHALACROCORAX

   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int run(
      std::shared_ptr<FileManager const> fileManager,
      ConnectionSpecs const &connection,
      PV_Init const &pv_init,
      std::string const &directory) {
   int status         = PV_SUCCESS;

   // Write a shared weights PVP file using the LocalPatchWeightsFile class, and then read it back
   // using primitive FileStream functions, and compare the result.
   fileManager->ensureDirectoryExists(directory); // path is relative to FileManager's baseDir.
   std::string testWritePath = directory + "/testWrite.pvp";

   int numArbors      = connection.getNumArbors();
   int nxp            = connection.getPatchSizeX();
   int nyp            = connection.getPatchSizeY();
   int nfp            = connection.getPatchSizeF();
   int nxPre          = connection.getNxGlobalRestrictedPre();
   int nyPre          = connection.getNyGlobalRestrictedPre();
   int nfPre          = connection.getNfPre();
   int nxPost         = connection.getNxGlobalRestrictedPost();
   int nyPost         = connection.getNyGlobalRestrictedPost();
   int xMargin        = requiredConvolveMargin(nxPre, nxPost, nxp, 'x', testWritePath.c_str());
   int yMargin        = requiredConvolveMargin(nyPre, nyPost, nyp, 'y', testWritePath.c_str());
   PVLayerLoc preLoc  =
        createLayerLoc(pv_init, nxPre, nyPre, nfPre, xMargin, yMargin, std::string("preLoc"));
   PVLayerLoc postLoc =
        createLayerLoc(pv_init, nxPost, nyPost, nfp, 0, 0, std::string("postLoc"));
   double timestamp;

   std::shared_ptr<WeightData> writeWeights =
         allocateWeights("writeWeights", numArbors, nxp, nyp, nfp, preLoc, postLoc);

   std::unique_ptr<LocalPatchWeightsFile> wgtFile(new LocalPatchWeightsFile(
      fileManager,
      testWritePath,
      writeWeights,
      &preLoc,
      &postLoc,
      true /*fileExtendedFlag*/,
      false /*compressedFlag*/,
      false /*readOnlyFlag*/,
      false /*clobberFlag*/,
      false /*verifyWrites*/));


   timestamp = 10.0;
   setWeights1(writeWeights, preLoc, postLoc);
   wgtFile->write(timestamp);

   timestamp = 15.0;
   setWeights2(writeWeights, preLoc, postLoc);
   wgtFile->write(timestamp);

   wgtFile = std::unique_ptr<LocalPatchWeightsFile>();

   // Now, read the weights back, without using LocalPatchWeightsFile, and compare the results to
   // weights1 and weights2

   // If a process has MPI batch index != 0, there might not be a weights file in the block for
   // this process, if the process sees the same file system as that with MPI batch index == 0.

   std::shared_ptr<FileStream> checkWriteFile = nullptr;

   if (fileManager->isRoot()) {
      bool wgtFileExistsFlag = fileManager->queryFileExists(testWritePath);
      std::string checkWritePath;
      if (wgtFileExistsFlag) {
         checkWritePath = fileManager->convertToEffectivePath(testWritePath);
      }
      else {
         auto baseDirectory = fileManager->getBaseDirectory();
         auto mpiBlock = fileManager->getMPIBlock();
         int col = mpiBlock->getStartColumn() / mpiBlock->getNumColumns();
         int row = mpiBlock->getStartRow() / mpiBlock->getNumRows();
         checkWritePath = FileManager::createBlockDirNameFromColRowElem(baseDirectory, col, row, 0);
         checkWritePath.append(testWritePath);
      }
      InfoLog().printf(
            "<LocalPatchWeightsFileTest.cpp:%d> Reading %s\n", __LINE__, checkWritePath.c_str());
      checkWriteFile = std::make_shared<FileStream>(
            checkWritePath.c_str(), std::ios_base::in | std::ios_base::binary);
   }
   if (status == PV_SUCCESS) {
      setWeights1(writeWeights, preLoc, postLoc);
      auto checkWeights1 = readFromFileStream(checkWriteFile, 0/*frame number*/, fileManager);
      status = compareWeights(
            writeWeights /*expected*/, checkWeights1 /*observed*/,
            preLoc.nx, preLoc.ny, postLoc.nx, postLoc.ny,
            std::string(directory + " write test, frame 0"));
   }
   if (status == PV_SUCCESS) {
      setWeights2(writeWeights, preLoc, postLoc);
      auto checkWeights2 = readFromFileStream(checkWriteFile, 1/*frame number*/, fileManager);
      status = compareWeights(
            writeWeights /*expected*/, checkWeights2 /*observed*/,
            preLoc.nx, preLoc.ny, postLoc.nx, postLoc.ny,
            std::string(directory + " write test, frame 1"));
   }
   if (status != PV_SUCCESS) { return EXIT_FAILURE; }

#ifdef PHALACROCORAX
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
   std::shared_ptr<WeightData> readWeights =
         allocateWeights("readWeights", numArbors, nxp, nyp, nfp, preLoc, postLoc);
   wgtFile = std::unique_ptr<LocalPatchWeightsFile>(new LocalPatchWeightsFile(
      fileManager,
      testReadPath,
      readWeights,
      &preLoc,
      &postLoc,
      true /*fileExtendedFlag*/,
      false /*compressedFlag*/,
      true /*readOnlyFlag*/,
      false /*clobberFlag*/,
      false /*verifyWrites*/));
   double readTimestamp3;
   if (status == PV_SUCCESS) {
      wgtFile->read(readTimestamp3);
      status = compareWeights(
            weights3 /*expected*/, readWeights /*observed*/,
            preLoc.nx, preLoc.ny, postLoc.nx, postLoc.ny,
            std::string(directory + " read test, frame 0"));
   }
   if (status == PV_SUCCESS) {
      if (readTimestamp3 != timestamp3) {
         ErrorLog().printf("%s read test, frame 0, expected timestamp %f, received %f\n",
               directory.c_str(), timestamp3, readTimestamp3);
         status = PV_FAILURE;
      }
   }
   double readTimestamp4;
   if (status == PV_SUCCESS) {
      wgtFile->read(readTimestamp4);
      status = compareWeights(
            weights4 /*expected*/, readWeights /*observed*/,
            preLoc.nx, preLoc.ny, postLoc.nx, postLoc.ny,
            std::string(directory + " read test, frame 1"));
   }
   if (status == PV_SUCCESS) {
      if (readTimestamp4 != timestamp4) {
         ErrorLog().printf("%s read test, frame 1, expected timestamp %f, received %f\n",
               directory.c_str(), timestamp4, readTimestamp4);
         status = PV_FAILURE;
      }
   }
#endif // PHALACROCORAX

   if (status == PV_SUCCESS) {
      InfoLog() << "Test passed.\n";
   }
   else {
      Fatal() << "Test failed.\n";
   }
   return status;
}

std::shared_ptr<WeightData> allocateWeights(
      std::string const &name,
      int numArbors, int nxp, int nyp, int nfp,
      PVLayerLoc const &preLoc, PVLayerLoc const &postLoc) {
   int nxPreRestricted = preLoc.nx;
   int nxPost          = postLoc.nx;
   int xMargin         = requiredConvolveMargin(nxPreRestricted, nxPost, nxp, 'x', "Connection");
   int nxPreExtended   = nxPreRestricted + 2 * xMargin;
   int nyPreRestricted = preLoc.ny;
   int nyPost          = postLoc.ny;
   int yMargin         = requiredConvolveMargin(nyPreRestricted, nyPost, nyp, 'y', "Connection");
   int nyPreExtended   = nyPreRestricted + 2 * yMargin;
   auto weightData     = std::make_shared<WeightData>(
         name, numArbors, nxp, nyp, nfp, nxPreExtended, nyPreExtended, preLoc.nf);
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
      if (status) {
         ErrorLog().printf("Unable to stat \"%s\": %s\n", path.c_str(), strerror(errno));
         return PV_FAILURE;
      }
      auto dirContents = fileManager->listDirectory(path);
      for (auto &d : dirContents) {
         std::string dirEntry(path + "/" + d);
         status = fileManager->stat(dirEntry, statbuf);
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
      std::shared_ptr<WeightData const> expectedWeights,
      std::shared_ptr<WeightData const> observedWeights,
      int nxRestrictedPre, int nyRestrictedPre, int nxRestrictedPost, int nyRestrictedPost,
      std::string const &label) {
   int status = PV_SUCCESS;
   if (expectedWeights->getNumArbors() != observedWeights->getNumArbors()) {
      ErrorLog().printf(
            "compareWeights, %s: numbers of arbors differ (%d versus %d)\n",
            label.c_str(), expectedWeights->getNumArbors(), observedWeights->getNumArbors());
      status = PV_FAILURE;
   }
   int numArbors = expectedWeights->getNumArbors();

   if (expectedWeights->getPatchSizeX() != observedWeights->getPatchSizeX()) {
      ErrorLog().printf(
            "compareWeights, %s: PatchSizeX differs (%d versus %d)\n",
            label.c_str(), expectedWeights->getPatchSizeX(), observedWeights->getPatchSizeX());
      status = PV_FAILURE;
   }
   int patchSizeX = expectedWeights->getPatchSizeX();
   if (expectedWeights->getPatchSizeY() != observedWeights->getPatchSizeY()) {
      ErrorLog().printf(
            "compareWeights, %s: PatchSizeY differs (%d versus %d)\n",
            label.c_str(), expectedWeights->getPatchSizeY(), observedWeights->getPatchSizeY());
      status = PV_FAILURE;
   }
   int patchSizeY = expectedWeights->getPatchSizeY();
   if (expectedWeights->getPatchSizeF() != observedWeights->getPatchSizeF()) {
      ErrorLog().printf(
            "compareWeights, %s: PatchSizeF differs (%d versus %d)\n",
            label.c_str(), expectedWeights->getPatchSizeF(), observedWeights->getPatchSizeF());
      status = PV_FAILURE;
   }
   int patchSizeF = expectedWeights->getPatchSizeF();

   int xMargin = requiredConvolveMargin(
         nxRestrictedPre, nxRestrictedPost, patchSizeX, 'x', "compareWeights");
   int yMargin = requiredConvolveMargin(
         nyRestrictedPre, nyRestrictedPost, patchSizeY, 'y', "compareWeights");
   if (expectedWeights->getNumDataPatchesX() < nxRestrictedPre + 2 * xMargin) {
      ErrorLog().printf(
            "compareWeights, %s: expectedWeights does not have enough patches in the x-direction "
            "(nxRestricted = %d, required margins %d, but "
            "expectedWeights is only %d patches wide)\n",
            label.c_str(), nxRestrictedPre, xMargin, expectedWeights->getNumDataPatchesX());
      status = PV_FAILURE;
   }
   if (expectedWeights->getNumDataPatchesY() < nyRestrictedPre + 2 * yMargin) {
      ErrorLog().printf(
            "compareWeights, %s: expectedWeights does not have enough patches in the y-direction "
            "(nyRestricted = %d, required margins %d, but "
            "expectedWeights is only %d patches high)\n",
            label.c_str(), nyRestrictedPre, yMargin, expectedWeights->getNumDataPatchesY());
      status = PV_FAILURE;
   }
   if (observedWeights->getNumDataPatchesX() < nxRestrictedPre + 2 * xMargin) {
      ErrorLog().printf(
            "compareWeights, %s: observedWeights does not have enough patches in the x-direction "
            "(nxRestricted = %d, required margins %d, but observedWeights is only %d patches wide)\n",
            label.c_str(), nxRestrictedPre, xMargin, observedWeights->getNumDataPatchesX());
      status = PV_FAILURE;
   }
   if (observedWeights->getNumDataPatchesY() < nyRestrictedPre + 2 * yMargin) {
      ErrorLog().printf(
            "compareWeights, %s: observedWeights does not have enough patches in the y-direction "
            "(nyRestricted = %d, required margins %d, but observedWeights is only %d patches high)\n",
            label.c_str(), nyRestrictedPre, yMargin, observedWeights->getNumDataPatchesY());
      status = PV_FAILURE;
   }
   if (expectedWeights->getNumDataPatchesF() != observedWeights->getNumDataPatchesF()) {
      ErrorLog().printf(
            "compareWeights, %s: NumDataPatchesF differs (%d versus %d)\n",
            label.c_str(),
            expectedWeights->getNumDataPatchesF(),
            observedWeights->getNumDataPatchesF());
      status = PV_FAILURE;
   }
   int nf = expectedWeights->getNumDataPatchesF();

   int xStartIndex1 = (expectedWeights->getNumDataPatchesX() - nxRestrictedPre) / 2;
   int yStartIndex1 = (expectedWeights->getNumDataPatchesY() - nyRestrictedPre) / 2;
   int xStartIndex2 = (observedWeights->getNumDataPatchesX() - nxRestrictedPre) / 2;
   int yStartIndex2 = (observedWeights->getNumDataPatchesY() - nyRestrictedPre) / 2;

   for (int a = 0; a < numArbors; ++a) {
      for (int y = 0; y < nyRestrictedPre + 2 * yMargin; ++y) {
         for (int x = 0; x < nxRestrictedPre + 2 * xMargin; ++x) {
            for (int f = 0; f < nf; ++f) {
               float const *expectedPatch =
                     expectedWeights->getDataFromXYF(a, x + xStartIndex1, y + yStartIndex1, f);
               float const *observedPatch =
                     observedWeights->getDataFromXYF(a, x + xStartIndex2, y + yStartIndex2, f);
               // Need to compute valid region; should be same for expected and observed weights
               int xPatchDim, xPatchStart, yPatchDim, yPatchStart;
               PatchGeometry::calcPatchData(
                     x + xStartIndex1,
                     nxRestrictedPre,
                     xStartIndex1,
                     xStartIndex1,
                     nxRestrictedPost,
                     0, 0,
                     patchSizeX,
                     &xPatchDim, &xPatchStart, nullptr, nullptr, nullptr);
               PatchGeometry::calcPatchData(
                     y + yStartIndex1,
                     nyRestrictedPre,
                     yStartIndex1,
                     yStartIndex1,
                     nyRestrictedPost,
                     0, 0,
                     patchSizeY,
                     &yPatchDim, &yPatchStart, nullptr, nullptr, nullptr);
               for (int ky = yPatchStart; ky < yPatchStart + yPatchDim; ++ky) {
                  for (int kx = xPatchStart; kx < xPatchStart + xPatchDim; ++kx) {
                     for (int kf = 0; kf < patchSizeF; ++kf) {
                        long index = kIndex(kx, ky, kf, patchSizeX, patchSizeY, patchSizeF);
                        float discrepancy = observedPatch[index] - expectedPatch[index];
                        if (std::abs(discrepancy) > tolerance * std::abs(expectedPatch[index])) {
                           ErrorLog().printf(
                                 "compareWeights, %s: weights do not agree at patch with "
                                 "arbor %d, restricted index x=%d, y=%d, f=%d, "
                                 "patch element at x=%d, y=%d, f=%d "
                                 "expected %f, observed %f, discrepancy %g, relative error %g)\n",
                                 label.c_str(), a, x, y, f, kx, ky, kf,
                                 (double)expectedPatch[index], (double)observedPatch[index],
                                 (double)discrepancy,
                                 (double)std::abs(discrepancy/expectedPatch[index]));
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
   loc.bcast   = 0;
   loc.halo.lt = xMargin;
   loc.halo.rt = xMargin;
   loc.halo.dn = yMargin;
   loc.halo.up = yMargin;
   return loc;
}

std::shared_ptr<WeightData> createWgts3(
      int numArbors, int nxp, int nyp, int nfp,
      PVLayerLoc const &preLoc, PVLayerLoc const &postLoc) {
   auto weightData       = allocateWeights("createWgts3", numArbors, nxp, nyp, nfp, preLoc, postLoc);
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
         long kGlobal          = kIndex(xGlobal, yGlobal, fIndex, nxGlobalExt, nyGlobalExt, nf);
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
   auto weightData = allocateWeights("createWgts4", numArbors, nxp, nyp, nfp, preLoc, postLoc);

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
         long kGlobal          = kIndex(xGlobal, yGlobal, fIndex, nxGlobalExt, nyGlobalExt, nf);
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
      long filePos = fileStream->getInPos();
      InfoLog().printf(
            "Reading header from \"%s\", position %ld, length %zu\n",
            fileStream->getFileName().c_str(), filePos, sizeof(header));
      fileStream->read(&header, static_cast<long>(sizeof(header)));

      for (int f = 0; f < frameNumber; ++f) {
         int patchSize = 8L + header.nxp * header.nyp * header.nfp * header.baseHeader.dataSize;
         long frameDataSize =
               static_cast<long>(patchSize * header.numPatches * header.baseHeader.numRecords);
         fileStream->setInPos(frameDataSize, std::ios_base::cur);
         filePos = fileStream->getInPos();
         InfoLog().printf(
               "Reading header from \"%s\", position %ld, length %zu\n",
               fileStream->getFileName().c_str(), filePos, sizeof(header));
         fileStream->read(&header, static_cast<long>(sizeof(header)));
      }
   }
   MPI_Bcast(&header, sizeof(header), MPI_BYTE, rootProc, mpiBlock->getComm());
   int numArbors  = header.baseHeader.numRecords;
   int blockNxExt = header.baseHeader.nxExtended;
   int blockNyExt = header.baseHeader.nyExtended;
   int nfPre      = header.baseHeader.nf;
   int patchSize  = header.nxp * header.nyp * header.nfp;
   FatalIf(
         header.baseHeader.dataSize != static_cast<int>(sizeof(float)), // TODO: compressed
         "header dataSize is %d instead of %d. File \"%s\", frameNumber %d\n",
         header.baseHeader.dataSize,
         static_cast<int>(sizeof(float)),
         fileStream ? fileStream->getFileName().c_str() : "(null)",
         frameNumber);
   long patchSizeBytes  = static_cast<long>(patchSize * header.baseHeader.dataSize);
   long numPatches      = static_cast<long>(blockNxExt * blockNyExt * nfPre);
   auto blockWeightData = std::make_shared<WeightData>(
         "blockWeightData",
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
         "localWeightData",
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
      FatalIf(
            static_cast<int>(blockArborSizeBytes) != blockArborSizeBytes,
            "blockArborSizeBytes = %ld is too big for MPI_Bcast()\n",
            blockArborSizeBytes);
      float *blockArborPointer = blockWeightData->getData(a);
      MPI_Bcast(
            blockArborPointer, (int)blockArborSizeBytes, MPI_BYTE, rootProc, mpiBlock->getComm());

      // Each process extracts its part of the weights from the entire block
      int xStart    = localNx * mpiBlock->getColumnIndex();
      int yStart    = localNy * mpiBlock->getRowIndex();
      int lineSize  = localNxExt * nfPre * patchSize;
      for (int ky = 0; ky < localNyExt; ++ky) {
         long blockStartIndex = kIndex(xStart, yStart + ky, 0, blockNxExt, blockNyExt, nfPre);
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
      BufferUtils::WeightHeader weightHeader;
      weightHeader.baseHeader.headerSize = 4 * NUM_WGT_PARAMS;
      weightHeader.baseHeader.numParams = NUM_WGT_PARAMS;
      weightHeader.baseHeader.fileType = PVP_WGT_FILE_TYPE;
      weightHeader.baseHeader.nx = preLayerLoc.nx * mpiBlock->getNumColumns();
      weightHeader.baseHeader.ny = preLayerLoc.ny * mpiBlock->getNumRows();
      weightHeader.baseHeader.nf = preLayerLoc.nf;
      weightHeader.baseHeader.numRecords = numArbors;
      weightHeader.baseHeader.recordSize = 0;
      weightHeader.baseHeader.dataSize = static_cast<int>(sizeof(float));
      weightHeader.baseHeader.dataType = BufferUtils::HeaderDataTypeEnum::FLOAT;
      weightHeader.baseHeader.nxProcs = 1;
      weightHeader.baseHeader.nyProcs = 1;

      int marginX = requiredConvolveMargin(
            preLayerLoc.nx, postLayerLoc.nx, nxp, 'x', fileStream->getFileName().c_str());
      int nxExtended = weightHeader.baseHeader.nx + marginX + marginX;
      weightHeader.baseHeader.nxExtended = nxExtended;

      int marginY = requiredConvolveMargin(
            preLayerLoc.ny, postLayerLoc.ny, nyp, 'y', fileStream->getFileName().c_str());
      int nyExtended = weightHeader.baseHeader.ny + marginY + marginY;
      weightHeader.baseHeader.nyExtended = nyExtended;

      weightHeader.baseHeader.kx0 = 0;
      weightHeader.baseHeader.ky0 = 0;
      weightHeader.baseHeader.nBatch = 1;
      weightHeader.baseHeader.nBands = numArbors;
      weightHeader.baseHeader.timestamp = timestamp;
      weightHeader.nxp = nxp;
      weightHeader.nyp = nyp;
      weightHeader.nfp = nfp;
      weightHeader.minVal = minVal;
      weightHeader.maxVal = maxVal;
      weightHeader.numPatches = nxExtended * nyExtended * weightHeader.baseHeader.nf;

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
                  long lineStartIndexInBlock = kIndex(
                        mpiColumn * preLayerLoc.nx,
                        mpiRow * preLayerLoc.ny + y,
                        0,
                        weightHeader.baseHeader.nxExtended,
                        weightHeader.baseHeader.nyExtended,
                        weightHeader.baseHeader.nf);
                  long lineStartFileOffset = patchSizeInFile * lineStartIndexInBlock;
                  long lineStartFilePos = arborStartInFile + lineStartFileOffset;
                  fileStream->setOutPos(lineStartFilePos, std::ios_base::beg);
                  for (int x = 0; x < nxLocalExtended; ++x) {
                     for (int f = 0; f < nf; ++f) {
                        fileStream->write(&patchHeader, patchHeaderSize);
                        long patchIndex = kIndex(x, y, f, nxLocalExtended, nyLocalExtended, nf);
                        long dataIndex = patchIndex * patchSizeOverall;
                        float *patchData = &mpiBuffer.at(dataIndex);
                        fileStream->write(patchData, patchSizeBytes);
                     } // f
                  } // x
               } // y
            } // a
         } // mpiColumn
      } // mpiRow
   } // if isRoot()
   else if (mpiBlock->getBatchIndex() == 0) {
      for (int a = 0; a < numArbors; ++a) {
         MPI_Send(weightData->getData(a), static_cast<int>(bufferSize), MPI_FLOAT,
         0 /*receiving rank*/, 140 + a /*tag*/, mpiBlock->getComm());
      }  
   }
}

void setWeights1(
      std::shared_ptr<WeightData> weightData,
      PVLayerLoc const &preLoc,
      PVLayerLoc const &postLoc) {
   int numArbors = weightData->getNumArbors();
   int nxLocalExt        = preLoc.nx + preLoc.halo.lt + preLoc.halo.rt;
   int nyLocalExt        = preLoc.ny + preLoc.halo.dn + preLoc.halo.up;
   int nf                = preLoc.nf;
   int nxGlobalExt       = preLoc.nxGlobal + preLoc.halo.lt + preLoc.halo.rt;
   int nyGlobalExt       = preLoc.nyGlobal + preLoc.halo.dn + preLoc.halo.up;
   long numPatchesGlobal = static_cast<long>(nxGlobalExt * nyGlobalExt * nf);
   long numPatchesLocal  = static_cast<long>(nxLocalExt * nyLocalExt * nf); 
   long patchSizeOverall = static_cast<long>(weightData->getPatchSizeOverall());
   for (int a = 0; a < numArbors; ++a) {
      for (long k = 0; k < numPatchesLocal; ++k) {
         int xLocal            = kxPos(k, nxLocalExt, nyLocalExt, nf);
         int yLocal            = kyPos(k, nxLocalExt, nyLocalExt, nf);
         int fIndex            = featureIndex(k, nxLocalExt, nyLocalExt, nf);
         int xGlobal           = xLocal + preLoc.kx0;
         int yGlobal           = yLocal + preLoc.ky0;
         long kGlobal          = kIndex(xGlobal, yGlobal, fIndex, nxGlobalExt, nyGlobalExt, nf);
         long baseIndexGlobal  = patchSizeOverall * (a * numPatchesGlobal + kGlobal);
         float *patchLocation  = weightData->getDataFromDataIndex(a, k);
         for (long p = 0; p < patchSizeOverall; ++p) {
            float indexGlobal = static_cast<float>(baseIndexGlobal + p + 1); 
            patchLocation[p] = indexGlobal;
         }
      }
   }
}

void setWeights2(
      std::shared_ptr<WeightData> weightData,
      PVLayerLoc const &preLoc,
      PVLayerLoc const &postLoc) {
   int numArbors = weightData->getNumArbors();
   int nxLocalExt        = preLoc.nx + preLoc.halo.lt + preLoc.halo.rt;
   int nyLocalExt        = preLoc.ny + preLoc.halo.dn + preLoc.halo.up;
   int nf                = preLoc.nf;
   int nxGlobalExt       = preLoc.nxGlobal + preLoc.halo.lt + preLoc.halo.rt;
   int nyGlobalExt       = preLoc.nyGlobal + preLoc.halo.dn + preLoc.halo.up;
   long numPatchesGlobal = static_cast<long>(nxGlobalExt * nyGlobalExt * nf);
   long numPatchesLocal  = static_cast<long>(nxLocalExt * nyLocalExt * nf); 
   long patchSizeOverall = static_cast<long>(weightData->getPatchSizeOverall());
   for (int a = 0; a < numArbors; ++a) {
      for (long k = 0; k < numPatchesLocal; ++k) {
         int xLocal            = kxPos(k, nxLocalExt, nyLocalExt, nf);
         int yLocal            = kyPos(k, nxLocalExt, nyLocalExt, nf);
         int fIndex            = featureIndex(k, nxLocalExt, nyLocalExt, nf);
         int xGlobal           = xLocal + preLoc.kx0;
         int yGlobal           = yLocal + preLoc.ky0;
         long kGlobal          = kIndex(xGlobal, yGlobal, fIndex, nxGlobalExt, nyGlobalExt, nf);
         long baseIndexGlobal  = patchSizeOverall * (a * numPatchesGlobal + kGlobal);
         float *patchLocation  = weightData->getDataFromDataIndex(a, k);
         for (long p = 0; p < patchSizeOverall; ++p) {
            float indexGlobal = static_cast<float>(baseIndexGlobal + p + 1); 
            patchLocation[p] = std::sqrt(indexGlobal);
         }
      }
   }
}
