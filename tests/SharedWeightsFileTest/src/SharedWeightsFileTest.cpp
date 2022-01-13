/*
 * SharedWeightsFileTest.cpp
 *
 */

#include <columns/PV_Init.hpp>
#include <io/FileManager.hpp>
#include <io/SharedWeightsFile.hpp>
#include <structures/MPIBlock.hpp>
#include <utils/BufferUtilsPvp.hpp>

#include <algorithm> // used by calcMinVal and calcMaxVal
#include <memory>
#include <limits> // used by calcMinVal and calcMaxVal

using namespace PV;

float calcMinVal(std::shared_ptr<WeightData const> weightData);
float calcMaxVal(std::shared_ptr<WeightData const> weightData);

// Recursively deletes the contents of the directory specified by path, and removes the directory
// itself, unless path is "." or ends in "/."
int cleanDirectory(std::shared_ptr<FileManager const> fileManager, std::string const &path);

int compareWeights(
      std::shared_ptr<WeightData const> weights1, std::shared_ptr<WeightData const> weights2);

std::shared_ptr<WeightData> createWgts1(
      int numArbors, int nxp, int nyp, int nfp, int nxPre, int nyPre, int nfPre);
std::shared_ptr<WeightData> createWgts2(
      int numArbors, int nxp, int nyp, int nfp, int nxPre, int nyPre, int nfPre);
std::shared_ptr<WeightData> createWgts3(
      int numArbors, int nxp, int nyp, int nfp, int nxPre, int nyPre, int nfPre);
std::shared_ptr<WeightData> createWgts4(
      int numArbors, int nxp, int nyp, int nfp, int nxPre, int nyPre, int nfPre);

std::shared_ptr<FileManager> createFileManager(PV_Init &pv_init_obj);

std::shared_ptr<WeightData> readFromFileStream(
      std::shared_ptr<FileStream> fileStream, int frameNumber, std::shared_ptr<FileManager const> fileManager);
void writeToFileStream(
      std::shared_ptr<FileStream> &fileStream,
      std::shared_ptr<WeightData const> weightData,
      double timestamp,
      std::shared_ptr<FileManager const> fileManager);

int main(int argc, char *argv[]) {
   int status = PV_SUCCESS;

   PV_Init pv_init_obj(&argc, &argv, false /* do not allow extra arguments */);
   std::shared_ptr<FileManager> fileManager = createFileManager(pv_init_obj);
   fileManager->ensureDirectoryExists("."); // "." is relative to FileManager's baseDir.

   // Delete contents of old output directory, to start with a clean slate.
   cleanDirectory(fileManager, std::string("."));

   int numArbors = 2;
   int nxp       = 5;
   int nyp       = 7;
   int nfp       = 3;
   int nxPre     = 2;
   int nyPre     = 2;
   int nfPre     = 4;
   double timestamp;

   // Write a shared weights PVP file using the SharedWeightsFile class, and then read it back
   // using primitive FileStream functions, and compare the result. 
   std::string testWritePath("testWeightsWrite.pvp");
   auto weights1 = createWgts1(numArbors, nxp, nyp, nfp, nxPre, nyPre, nfPre);

   std::unique_ptr<SharedWeightsFile> wgtFile(new SharedWeightsFile(
      fileManager,
      testWritePath,
      weights1,
      false /*compressedFlag*/,
      false /*readOnlyFlag*/,
      false /*verifyWrites*/));

   timestamp = 10.0;
   wgtFile->write(*weights1, timestamp);

   auto weights2 = createWgts2(numArbors, nxp, nyp, nfp, nxPre, nyPre, nfPre);
   timestamp = 15.0;
   wgtFile->write(*weights2, timestamp);

   wgtFile = std::unique_ptr<SharedWeightsFile>();

   // Now, read the weights back, without using SharedWeightsFile, and compare the results to
   // weights1 and weights2
   auto checkWriteFile = fileManager->open(testWritePath, std::ios_base::in | std::ios_base::binary);
   if (status == PV_SUCCESS) {
      auto checkWeights1 = readFromFileStream(checkWriteFile, 0/*frame number*/, fileManager);
      status = compareWeights(weights1, checkWeights1);
   }
   if (status == PV_SUCCESS) {
      auto checkWeights2 = readFromFileStream(checkWriteFile, 1/*frame number*/, fileManager);
      status = compareWeights(weights2, checkWeights2);
   }
   if (status != PV_SUCCESS) { return EXIT_FAILURE; }

   // Write a shared weights PVP file using primitive FileStream functions, and then read it back
   // using the SharedWeightsFile class, and compare the result.
   //
   auto weights3 = createWgts3(numArbors, nxp, nyp, nfp, nxPre, nyPre, nfPre);
   auto weights4 = createWgts4(numArbors, nxp, nyp, nfp, nxPre, nyPre, nfPre);
   std::string testReadPath("testWeightsRead.pvp");
   // File shouldn't exist; create it.
   auto testReadFile = fileManager->open(testReadPath, std::ios_base::out);
   auto mode = std::ios_base::in | std::ios_base::out | std::ios_base::binary;
   testReadFile = fileManager->open(testReadPath, mode); // closes & reopens with read/write mode
   double timestamp3 = 20.0;
   writeToFileStream(testReadFile, weights3, timestamp3, fileManager);
   double timestamp4 = 25.0;
   writeToFileStream(testReadFile, weights4, timestamp4, fileManager);
   testReadFile = nullptr; // closes file

   // Now read the weights using the SharedWeightsFile class, and compare the results
   wgtFile = std::unique_ptr<SharedWeightsFile>(new SharedWeightsFile(
      fileManager,
      testReadPath,
      weights3,
      false /*compressedFlag*/,
      true /*readOnlyFlag*/,
      false /*verifyWrites*/));
   auto readWeights3 = std::make_shared<WeightData>(
      numArbors,
      nxp, nyp, nfp,
      nxPre, nyPre, nfPre);
   double readTimestamp3;
   wgtFile->read(*readWeights3, readTimestamp3);

   if (status == PV_SUCCESS) {
      InfoLog() << "Test passed.\n";
   }
   else {
      Fatal() << "Test failed.\n";
   }

   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
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
      std::shared_ptr<WeightData const> weights1, std::shared_ptr<WeightData const> weights2) {
   int status = PV_SUCCESS;
   if (weights1->getNumArbors() != weights2->getNumArbors()) {
      ErrorLog().printf(
            "compareWeights: numbers of arbors differ (%d versus %d)\n",
            weights1->getNumArbors(), weights2->getNumArbors());
      status = PV_FAILURE;
   }
   if (weights1->getPatchSizeX() != weights2->getPatchSizeX()) {
      ErrorLog().printf(
            "compareWeights: PatchSizeX differs (%d versus %d)\n",
            weights1->getPatchSizeX(), weights2->getPatchSizeX());
      status = PV_FAILURE;
   }
   if (weights1->getPatchSizeY() != weights2->getPatchSizeY()) {
      ErrorLog().printf(
            "compareWeights: PatchSizeY differs (%d versus %d)\n",
            weights1->getPatchSizeY(), weights2->getPatchSizeY());
      status = PV_FAILURE;
   }
   if (weights1->getPatchSizeF() != weights2->getPatchSizeF()) {
      ErrorLog().printf(
            "compareWeights: PatchSizeF differs (%d versus %d)\n",
            weights1->getPatchSizeF(), weights2->getPatchSizeF());
      status = PV_FAILURE;
   }
   if (weights1->getNumDataPatchesX() != weights2->getNumDataPatchesX()) {
      ErrorLog().printf(
            "compareWeights: NumDataPatchesX differs (%d versus %d)\n",
            weights1->getNumDataPatchesX(), weights2->getNumDataPatchesX());
      status = PV_FAILURE;
   }
   if (weights1->getNumDataPatchesY() != weights2->getNumDataPatchesY()) {
      ErrorLog().printf(
            "compareWeights: NumDataPatchesY differs (%d versus %d)\n",
            weights1->getNumDataPatchesY(), weights2->getNumDataPatchesY());
      status = PV_FAILURE;
   }
   if (weights1->getNumDataPatchesF() != weights2->getNumDataPatchesF()) {
      ErrorLog().printf(
            "compareWeights: NumDataPatchesF differs (%d versus %d)\n",
            weights1->getNumDataPatchesF(), weights2->getNumDataPatchesF());
      status = PV_FAILURE;
   }
   return status;
}

std::shared_ptr<WeightData> createWgts1(
      int numArbors, int nxp, int nyp, int nfp, int nxPre, int nyPre, int nfPre) {
   auto weightData = std::make_shared<WeightData>(numArbors, nxp, nyp, nfp, nxPre, nyPre, nfPre);
   long elemsPerArbor = weightData->getPatchSizeOverall() * weightData->getNumDataPatchesOverall();
   pvAssert(elemsPerArbor == static_cast<long>(nxp * nyp * nfp * nxPre * nyPre * nfPre));
   for (int a = 0; a < numArbors; ++a) {
      float *arbor = weightData->getData(a);
      for (long k = 0; k < elemsPerArbor; ++k) {
         int index = a * elemsPerArbor + k;
         arbor[k] = static_cast<float>(index + 1);
      }
   }
   return weightData;
}

std::shared_ptr<WeightData> createWgts2(
      int numArbors, int nxp, int nyp, int nfp, int nxPre, int nyPre, int nfPre) {
   auto weightData = std::make_shared<WeightData>(numArbors, nxp, nyp, nfp, nxPre, nyPre, nfPre);
   long elemsPerArbor = weightData->getPatchSizeOverall() * weightData->getNumDataPatchesOverall();
   pvAssert(elemsPerArbor == static_cast<long>(nxp * nyp * nfp * nxPre * nyPre * nfPre));
   for (int a = 0; a < numArbors; ++a) {
      float *arbor = weightData->getData(a);
      for (long k = 0; k < elemsPerArbor; ++k) {
         int index = a * elemsPerArbor + k;
         arbor[k] = std::sqrt(static_cast<float>(index + 1));
      }
   }
   return weightData;
}

std::shared_ptr<WeightData> createWgts3(
      int numArbors, int nxp, int nyp, int nfp, int nxPre, int nyPre, int nfPre) {
   auto weightData = std::make_shared<WeightData>(numArbors, nxp, nyp, nfp, nxPre, nyPre, nfPre);
   long elemsPerArbor = weightData->getPatchSizeOverall() * weightData->getNumDataPatchesOverall();
   pvAssert(elemsPerArbor == static_cast<long>(nxp * nyp * nfp * nxPre * nyPre * nfPre));
   for (int a = 0; a < numArbors; ++a) {
      float *arbor = weightData->getData(a);
      for (long k = 0; k < elemsPerArbor; ++k) {
         int index = a * elemsPerArbor + k;
         arbor[k] = 1.0f - static_cast<float>(index)/static_cast<float>(numArbors * elemsPerArbor);
      }
   }
   return weightData;
}

std::shared_ptr<WeightData> createWgts4(
      int numArbors, int nxp, int nyp, int nfp, int nxPre, int nyPre, int nfPre) {
   auto weightData = std::make_shared<WeightData>(numArbors, nxp, nyp, nfp, nxPre, nyPre, nfPre);
   long elemsPerArbor = weightData->getPatchSizeOverall() * weightData->getNumDataPatchesOverall();
   pvAssert(elemsPerArbor == static_cast<long>(nxp * nyp * nfp * nxPre * nyPre * nfPre));
   for (int a = 0; a < numArbors; ++a) {
      float *arbor = weightData->getData(a);
      for (long k = 0; k < elemsPerArbor; ++k) {
         int index = a * elemsPerArbor + k;
         arbor[k] = -static_cast<float>(index + 1)/static_cast<float>(numArbors * elemsPerArbor);
      }
   }
   return weightData;
}

std::shared_ptr<FileManager> createFileManager(PV_Init &pv_init_obj) {
   auto mpiBlock  = pv_init_obj.getCommunicator()->getIOMPIBlock();
   auto arguments = pv_init_obj.getArguments();
   std::string baseDirectory = arguments->getStringArgument("OutputPath");
   FatalIf(baseDirectory.substr(0, 7) != "output/","OutputPath must begin with \"output\"\n");

   auto fileManager = std::make_shared<FileManager>(mpiBlock, baseDirectory);
   return fileManager;
}

std::shared_ptr<WeightData> readFromFileStream(
      std::shared_ptr<FileStream> fileStream, int frameNumber, std::shared_ptr<FileManager const> fileManager) {
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
   int numArbors = header.baseHeader.numRecords;
   int numPatchesX = header.baseHeader.nx;
   int numPatchesY = header.baseHeader.ny;
   int numPatchesF = header.baseHeader.nf;
   int patchSize  = header.nxp * header.nyp * header.nfp;
   pvAssert(header.baseHeader.dataSize == static_cast<int>(sizeof(float))); // TODO: compressed
   long patchSizeBytes = static_cast<long>(patchSize * header.baseHeader.dataSize);
   long numPatches = static_cast<long>(numPatchesX * numPatchesY * numPatchesF);
   std::vector<float> weightsVector(numPatches * patchSize);
   auto weightData = std::make_shared<WeightData>(
         numArbors, header.nxp, header.nyp, header.nfp, numPatchesX, numPatchesY, numPatchesF);
   for (int a = 0; a < numArbors; ++a) {
      if (fileManager->isRoot()) {
         for (long p = 0; p < numPatches; ++p) {
            float *patchAddress = weightData->getDataFromDataIndex(a, p);
            fileStream->setInPos(8L, std::ios_base::cur);
            fileStream->read(patchAddress, patchSizeBytes); 
         }
      }
      long arborSizeBytes = numPatches * patchSizeBytes;
      float *arborAddress = weightData->getData(a);
      MPI_Bcast(arborAddress, arborSizeBytes, MPI_BYTE, rootProc, mpiBlock->getComm());
   }
   return weightData;
}

void writeToFileStream(
      std::shared_ptr<FileStream> &fileStream,
      std::shared_ptr<WeightData const> weightData,
      double timestamp,
      std::shared_ptr<FileManager const> fileManager) {
   if (!fileManager->isRoot()) { return; }
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
   auto weightHeader = BufferUtils::buildSharedWeightHeader(
         nxp, nyp, nfp,
         numArbors,
         numPatchesX,
         numPatchesY,
         numPatchesF,
         timestamp,
         false /*compressFlag*/,
         minVal,
         maxVal);
   long const headerSize = 104L;
   FatalIf(
         static_cast<long>(sizeof(weightHeader)) != headerSize,
         "Weight header size should be 104 but is %zu\n",
         sizeof(weightHeader));
   fileStream->write(&weightHeader, headerSize);
   Patch patchHeader;
   patchHeader.nx     = static_cast<uint16_t>(nxp);
   patchHeader.ny     = static_cast<uint16_t>(nyp);
   patchHeader.offset = static_cast<uint32_t>(0);
   pvAssert(sizeof(patchHeader) == static_cast<std::size_t>(8));
   long patchSize = static_cast<long>(nxp * nyp * nfp);
   long patchSizeBytes = static_cast<long>(patchSize * weightHeader.baseHeader.dataSize);
   long numPatches = static_cast<long>(numPatchesX * numPatchesY * numPatchesF);
   for (int a = 0; a < numArbors; ++a) {
      float const *arbor = weightData->getData(a);
      for (long p = 0; p < numPatches; ++p) {
         long patchStart = p * patchSize;
         float const *patchAddress = weightData->getDataFromDataIndex(a, p);
         fileStream->write(&patchHeader, 8L);
         fileStream->write(patchAddress, patchSizeBytes); 
      }
   }
}
