/*
 * SparseBroadcastLayerFileTest.cpp
 *
 */

#include "columns/PV_Init.hpp"
#include "io/SparseBroadcastLayerFile.hpp"
#include "io/FileManager.hpp"
#include "io/FileStreamBuilder.hpp"
#include "structures/SparseList.hpp"
#include "utils/BufferUtilsMPI.hpp"     // gather, scatter
#include "utils/BufferUtilsPvp.hpp"     // struct ActivityHeader
#include "utils/PVAssert.hpp"           // pvAssert

#include <algorithm> // std::adjacent_find, std::copy, std::max
#include <cstdlib>   // EXIT_SUCCESS, EXIT_FAILURE
#include <ios>       // ios_base openmodes
#include <memory>    // shared_ptr
#include <string>    // std::string
#include <vector>    // std::vector

using namespace PV;

// Recursively deletes the contents of the directory specified by path, and removes the directory
// itself, unless path is "." or ends in "/."
int cleanDirectory(std::shared_ptr<FileManager const> fileManager, std::string const &path);

int compareLayerData(
      std::vector<SparseList<float>> const &expected,
      std::vector<SparseList<float>> const &observed);

std::vector<SparseList<float>> makeSparseBroadcastLayerData(
      std::shared_ptr<MPIBlock const> mpiBlock,
      int numFeatures,
      int globalBatchWidth,
      int start,
      int step,
      int batchStep);

// Read data from a layer .pvp file using only FileStream methods.
// On entry, layerData is a vector of the expected localBatchWidth
// (because of M-to-N, this may not the the batch width of the file).
// If timestampPtr is null, the timestamp of the file is ignored.
// If it is not null, the timestamp is filled with the file's timestamp
// for the specified index.
// The index argument is used the same way it is used in the
// SparseBroadcastLayerFile's getIndex() and setIndex() function members.
int readUsingFileStreamPrimitives(
      std::shared_ptr<FileManager const> fileManager,
      std::string const &path,
      std::vector<SparseList<float>> &layerData,
      double *timestampPtr,
      int index);

int runTests(std::shared_ptr<FileManager> fileManager);

BufferUtils::SparseFileTable setFilePositionFromIndex(
      FileStream &fileStream,
      std::shared_ptr<MPIBlock const> mpiBlock,
      int localBatchWidth,
      int index);

int testRead(
      std::shared_ptr<FileManager const> fileManager, int numFeatures, int globalBatchWidth);

int testReadMultipleFrames(
      std::shared_ptr<FileManager const> fileManager, int numFeatures, int globalBatchWidth);

int testReadRandomAccess(
      std::shared_ptr<FileManager const> fileManager, int numFeatures, int globalBatchWidth);

int testWrite(
      std::shared_ptr<FileManager const> fileManager, int numFeatures, int globalBatchWidth);

int testWriteMultipleFrames(
      std::shared_ptr<FileManager const> fileManager, int numFeatures, int globalBatchWidth);

int testTruncate(
      std::shared_ptr<FileManager const> fileManager, int numFeatures, int globalBatchWidth);

int writeUsingFileStreamPrimitives(
      std::shared_ptr<FileManager const> fileManager,
      std::string const &path,
      std::vector<SparseList<float>> layerData,
      double timestamp,
      int index);

int main(int argc, char *argv[]) {
   int status = PV_SUCCESS;

   PV_Init pv_init_obj(&argc, &argv, false /* do not allow extra arguments */);

   auto *communicator = pv_init_obj.getCommunicator();
   auto mpiBlock      = communicator->getIOMPIBlock();
   std::string baseDirectory = pv_init_obj.getStringArgument("OutputPath");
   FatalIf(
         baseDirectory.empty(),
         "OutputPath argument has not been set in config file.\n");
   auto fileManager = std::make_shared<FileManager>(mpiBlock, baseDirectory);

   // Delete contents of old output directory, to start with a clean slate.
   if (status == PV_SUCCESS) {
      status = cleanDirectory(fileManager, std::string("."));
   }
   if (status == PV_SUCCESS) {
      status = fileManager->makeDirectory(std::string("."));
      if (status != 0 and errno == EEXIST) {
         status = PV_SUCCESS;
      }
   }

   if (status == PV_SUCCESS) {
      status = runTests(fileManager);
   }
   if (status == PV_SUCCESS) {
      InfoLog() << "Test passed.\n";
   }
   else {
      Fatal() << "Test failed.\n";
   }

   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
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

int compareLayerData(
      std::vector<SparseList<float>> const &expected,
      std::vector<SparseList<float>> const &observed) {
   if (expected.size() != observed.size()) {
      ErrorLog() << "compareLayerData() called with expected batch width " << expected.size()
                 << " and observed batch width " << observed.size() << "\n";
      return PV_FAILURE;
   }
   int status = PV_SUCCESS;
   int localBatchWidth = static_cast<int>(expected.size());
   for (int b = 0; b < localBatchWidth; ++b) {
      auto const &expectedElem = expected[b];
      auto const &observedElem = observed[b];
      int numFeatures = expectedElem.getFeatures();
      if (observedElem.getFeatures() != numFeatures) {
         ErrorLog().printf(
               "compareLayerData(): "
               "batch element %d expected to have %d features but has %d features.\n",
               b, numFeatures, observedElem.getFeatures());
         status = PV_FAILURE;
         continue;
      }
      std::vector<SparseList<float>::Entry> observedList = observedElem.getContents();
      std::vector<SparseList<float>::Entry> expectedList = expectedElem.getContents();
      if (observedList.size() != expectedList.size()) {
         ErrorLog().printf(
               "compareLayerData(): batch element %d expected to have %d nonzero entries "
               "but has %d nonzero entries.\n",
               b, expectedList.size(), observedList.size());
         status = PV_FAILURE;
         continue;
      }
      auto numEntries = static_cast<std::size_t>(expectedList.size());
      for (std::size_t n = 0; n < numEntries; ++n) {
         SparseList<float>::Entry observedEntry = observedList[n];
         SparseList<float>::Entry expectedEntry = expectedList[n];
         if (observedEntry.index != expectedEntry.index) {
            ErrorLog().printf(
                  "compareLayerData() batch element %d, entry %zu: "
                  "expected index " PRIu32 ", observed " PRIu32 "\n",
                  b, n, expectedEntry.index, observedEntry.index);
            status = PV_FAILURE;
         }
         if (observedEntry.value != expectedEntry.value) {
            ErrorLog().printf(
                  "compareLayerData() batch element %d, entry %zu: "
                  "expected %f, observed %f (discrepancy %g)\n",
                  b, n,
                  static_cast<double>(expectedEntry.value),
                  static_cast<double>(observedEntry.value),
                  static_cast<double>(observedEntry.value - expectedEntry.value));
            status = PV_FAILURE;
         }
      }
   }
   return status;
}

std::vector<SparseList<float>> makeSparseBroadcastLayerData(
      std::shared_ptr<MPIBlock const> mpiBlock,
      int numFeatures,
      int globalBatchWidth,
      int start,
      int step,
      int batchStep) {
   int status = PV_SUCCESS;
   int localBatchWidth = globalBatchWidth / mpiBlock->getGlobalBatchDimension();
   pvAssert(localBatchWidth * mpiBlock->getGlobalBatchDimension() == globalBatchWidth);
   int kb0 = localBatchWidth *(mpiBlock->getStartBatch() + mpiBlock->getBatchIndex());
   std::vector<SparseList<float>> result(localBatchWidth);
   for (int b = 0; b < localBatchWidth; ++b) {
      int globalBatchIndex = b + kb0;
      int batchOffset = batchStep * globalBatchIndex;
      result[b].reset(1, 1, numFeatures);
      for (int f = start; f < numFeatures; f += step) {
         float value = static_cast<float>(start + step * f + batchOffset);
         result[b].addEntry(f, value);
      }
   }
   return result;
}

int readUsingFileStreamPrimitives(
      std::shared_ptr<FileManager const> fileManager,
      std::string const &path,
      std::vector<SparseList<float>> &layerData,
      double *timestampPtr,
      int index) {
   int status = PV_SUCCESS;
   int const sizeSparseListEntry = static_cast<int>(sizeof(SparseList<float>::Entry));
   auto fileStream = FileStreamBuilder(
      fileManager,
      path,
      false /*isTextFlag*/,
      true /*readOnlyFlag*/,
      false /*clobberFlag*/,
      false /*verifyWritesFlag*/).get();
   auto mpiBlock = fileManager->getMPIBlock();
   int numBatchProcs = mpiBlock->getBatchDimension();
   FatalIf(
         numBatchProcs <= 0,
         "MPIBlock's BatchDimension must be positive (value is %d)\n",
         numBatchProcs);
   int localBatchWidth = static_cast<int>(layerData.size());
   FatalIf(
         localBatchWidth <= 0,
         "readUsingFileStreamPrimitives() called with layerData vector of length %d\n",
         localBatchWidth);
   int expectedFileBatchWidth = localBatchWidth * numBatchProcs;
   std::vector<double> timestamps(expectedFileBatchWidth);

   if (fileStream) {
      // Read header
      fileStream->setInPos(0L, std::ios_base::beg);
      BufferUtils::ActivityHeader header;
      fileStream->read(&header, 80L);

      // Sanity check header values
      if (header.fileType != PVP_ACT_SPARSEVALUES_FILE_TYPE) {
         ErrorLog().printf(
               "readUsingFileStreamPrimitives(): file \"%s\" is not a sparse pvp file.\n",
               path.c_str());
         return PV_FAILURE;
      }
      if (header.nx != 1 or header.ny != 1) {
         ErrorLog().printf(
               "readUsingFileStreamPrimitives(): file \"%s\" is not a 1-by-1-by-nf pvp file.\n",
               path.c_str());
         return PV_FAILURE;
      }
      int numFramesInFile = header.nBands;
      if (numFramesInFile % expectedFileBatchWidth != 0) {
         ErrorLog().printf(
               "readUsingFileStreamPrimitives(): file \"%s\" has expected batch width of %d, "
               "but the header.nbands value %d is not a multiple of this number.\n",
               path.c_str(),
               expectedFileBatchWidth,
               header.nBands);
         return PV_FAILURE;
      }
      if (index * expectedFileBatchWidth > numFramesInFile) {
         ErrorLog().printf(
            "readUsingFileStreamPrimitives() called with index %d and expectedFileBatchWidth %d, "
            "but file \"%s\" only has %d PVP frames.\n",
            index,
            expectedFileBatchWidth,
            path.c_str(),
            numFramesInFile);
         return PV_FAILURE;
      }

      // Move to file position specified by index argument, and read data
      int numFeatures = header.nf;
      MPI_Bcast(&numFeatures, 1, MPI_INT, 0 /*root*/, mpiBlock->getComm());

      setFilePositionFromIndex(*fileStream, mpiBlock, localBatchWidth, index);
      std::vector<std::vector<SparseList<float>::Entry>> valuesFromFile(expectedFileBatchWidth);
      for (int b = 0; b < expectedFileBatchWidth; ++b) {
         fileStream->read(&timestamps.at(b), 8L);
         int numEntries;
         fileStream->read(&numEntries, 4L);
         valuesFromFile[b].resize(numEntries);
         long int sizeInBytes = numEntries * static_cast<long>(sizeof(SparseList<float>::Entry));
         fileStream->read(valuesFromFile[b].data(), sizeInBytes);
      }
      MPI_Bcast(
            timestamps.data(), expectedFileBatchWidth, MPI_DOUBLE, 0 /*root*/, mpiBlock->getComm());

      // Distribute values to rest of MPIBlock.
      int sendSize = localBatchWidth * numFeatures;
      std::vector<float> sendData(sendSize);
      for (int b = 0; b < localBatchWidth; ++b) {
         layerData[b].reset(1, 1, numFeatures);
         layerData[b].set(valuesFromFile[b]);
      }
      for (int r = 1; r < mpiBlock->getSize(); ++r) {
         int mpiBatchIndex = mpiBlock->calcBatchIndexFromRank(r);
         for (int b = 0; b < localBatchWidth; ++b) {
            int globalBatchIndex = b + localBatchWidth * mpiBatchIndex;
            MPI_Send(
                  valuesFromFile[globalBatchIndex].data(),
                  sizeSparseListEntry * static_cast<int>(valuesFromFile[globalBatchIndex].size()),
                  MPI_BYTE,
                  r,
                  345 + b /*tag*/,
                  mpiBlock->getComm());
            auto numValues = valuesFromFile[globalBatchIndex].size();
            for (decltype(numValues) iii = 0; iii < numValues; ++iii) {
               InfoLog().printf(
                     "Sending value %zu of %zu in batch element %d to rank %d: index %d, value %f\n",
                     iii, numValues, b, r,
                     valuesFromFile[globalBatchIndex][iii].index,
                     (double)(valuesFromFile[globalBatchIndex][iii].value));
            }
         }
      }
   }
   else {
      int numFeatures;
      MPI_Bcast(&numFeatures, 1, MPI_INT, 0 /*root*/, mpiBlock->getComm());
      MPI_Bcast(
            timestamps.data(), expectedFileBatchWidth, MPI_DOUBLE, 0 /*root*/, mpiBlock->getComm());
      for (int b = 0; b < localBatchWidth; ++b) {
         layerData[b].reset(1, 1, numFeatures);
         MPI_Status mpiStatus;
         MPI_Probe(0 /*source*/, 345 + b /*tag*/, mpiBlock->getComm(), &mpiStatus);
         int count;
         MPI_Get_count(&mpiStatus, MPI_BYTE, &count);
         int numEntries = count / sizeSparseListEntry;
         FatalIf(
               numEntries * sizeSparseListEntry != count,
               "readUsingFileStreamPrimitives received an MPI message of %d bytes, which is "
               "not a multiple of %d (the size of one sparse entry.\n",
               count, sizeSparseListEntry);
         std::vector<SparseList<float>::Entry> values(numEntries);
         MPI_Recv(
               values.data(),
               count,
               MPI_BYTE,
               0 /*source*/,
               345 + b /*tag*/,
               mpiBlock->getComm(),
               MPI_STATUS_IGNORE);
         layerData[b].set(values);
         for (int entry = 0; entry < numEntries; ++entry) {
            InfoLog().printf(
                  "Receiving value %d of %d in batch element %d on rank %d: index %d, value %f\n",
                  entry, numEntries, b, mpiBlock->getRank(),
                  values[entry].index,
                  (double)(values[entry].value));
         }
      }
   }

   // check timestamp consistency
   auto firstNonEqualPair =
         std::adjacent_find(timestamps.cbegin(), timestamps.cend(), std::not_equal_to<float>());
   if (firstNonEqualPair == timestamps.cend()) {
      if (timestampPtr) { *timestampPtr = timestamps.at(0); }
   }
   else {
      status = PV_FAILURE;
      ErrorLog().printf("timestamps did not agree across all batch elements\n");
   }
   return status;
}

int runTests(std::shared_ptr<FileManager> fileManager) {
   int status = PV_SUCCESS;
   int numFeatures = 32;
   int globalBatchWidth = 8;
   if (status == PV_SUCCESS) {
      status = testRead(fileManager, numFeatures, globalBatchWidth);
   }
   if (status == PV_SUCCESS) {
      status = testReadMultipleFrames(fileManager, numFeatures, globalBatchWidth);
   }
   if (status == PV_SUCCESS) {
      status = testReadRandomAccess(fileManager, numFeatures, globalBatchWidth);
   }
   if (status == PV_SUCCESS) {
      status = testWrite(fileManager, numFeatures, globalBatchWidth);
   }
   if (status == PV_SUCCESS) {
      status = testWriteMultipleFrames(fileManager, numFeatures, globalBatchWidth);
   }
   if (status == PV_SUCCESS) {
      status = testTruncate(fileManager, numFeatures, globalBatchWidth);
   }
   return status;
}

BufferUtils::SparseFileTable setFilePositionFromIndex(
      FileStream &fileStream,
      std::shared_ptr<MPIBlock const> mpiBlock,
      int localBatchWidth,
      int index) {
   int framesPerIndex = localBatchWidth * mpiBlock->getBatchDimension();
   int frameInFile = index * framesPerIndex;
   fileStream.setInPos(0L, std::ios_base::beg);
   BufferUtils::ActivityHeader header;
   fileStream.read(&header, 80L);
   int numBandsInFile = header.nBands;
   FatalIf(
         numBandsInFile < frameInFile,
         "setFilePositionFromIndex() called with index %d and %d frames per index, "
         "but \"%s\" has only %d frames.\n",
         index, framesPerIndex, fileStream.getFileName().c_str());
   BufferUtils::SparseFileTable sparseFileTable =
         BufferUtils::buildSparseFileTable(fileStream, numBandsInFile - 1);
   if (frameInFile == numBandsInFile) {
      fileStream.setInPos(0L, std::ios_base::end);
   }
   else {
      fileStream.setInPos(sparseFileTable.frameStartOffsets[frameInFile], std::ios_base::beg);
   }
   if (fileStream.writeable()) {
      fileStream.setOutPos(fileStream.getInPos(), std::ios_base::beg);
   }
   return sparseFileTable;
}

int testRead(
      std::shared_ptr<FileManager const> fileManager, int numFeatures, int globalBatchWidth) {
   int status = PV_SUCCESS;
   std::string filename("testRead.pvp");
   int start      = 1;
   int step       = 3;
   int batchStep  = 16;
   double timestamp = 11.0;
   auto mpiBlock    = fileManager->getMPIBlock();

   // Create test data and write it without using SparseBroadcastLayerFile
   auto layerData = makeSparseBroadcastLayerData(
         mpiBlock, numFeatures, globalBatchWidth, start, step, batchStep);
   writeUsingFileStreamPrimitives(fileManager, filename, layerData, timestamp, 0);

   // Read data back using SparseBroadcastLayerFile, and compare it to the test data
   int localBatchWidth = static_cast<int>(layerData.size());
   SparseBroadcastLayerFile readBackStream(
         fileManager,
         filename,
         numFeatures,
         localBatchWidth,
         true /*readOnlyFlag*/,
         false /*clobberFlag*/,
         false /*verifyWrites*/);
   std::vector<SparseList<float>> readBackData(localBatchWidth);
   for (int b = 0; b < localBatchWidth; ++b) {
      readBackData[b].reset(1, 1, numFeatures);
      readBackStream.setListLocation(&readBackData[b], b);
   }
   readBackStream.read();

   status = compareLayerData(layerData, readBackData);
   if (status != PV_SUCCESS) {
      ErrorLog().printf("testRead() failed.\n");
   }

   return status;
}

int testReadMultipleFrames(
      std::shared_ptr<FileManager const> fileManager, int numFeatures, int globalBatchWidth) {
   int status = PV_SUCCESS;
   auto mpiBlock = fileManager->getMPIBlock();
   std::string filename("testReadMultipleFrames.pvp");
   std::vector<double> timestamps{20.0, 22.0, 24.0, 26.0};
   std::vector<int> starts{10, 11, 12, 13};
   int step = 3;
   int batchStep = 16;

   // Make test data using FileStream primitive functions, without using SparseBroadcastLayerFile.
   std::vector<std::vector<SparseList<float>>> testData(4);
   for (int index = 0; index < 4; ++index) {
      testData[index] = makeSparseBroadcastLayerData(
            mpiBlock, numFeatures, globalBatchWidth, starts[index], step, batchStep);
      writeUsingFileStreamPrimitives(
            fileManager, filename, testData[index], timestamps[index], index);
   }

   // Read back the test data using SparseBroadcastLayerFile.
   int localBatchWidth = globalBatchWidth / mpiBlock->getGlobalBatchDimension();
   SparseBroadcastLayerFile readBackStream(
         fileManager,
         filename,
         numFeatures,
         localBatchWidth,
         true /*readOnlyFlag*/,
         false /*clobberFlag*/,
         false /*verifyWrites*/);
   std::vector<SparseList<float>> readBackData(localBatchWidth);
   for (int b = 0; b < localBatchWidth; ++b) {
      readBackData[b].reset(1, 1, numFeatures);
      readBackStream.setListLocation(&readBackData[b], b);
   }
   for (int index = 0; index < 4; ++index) {
      double timestamp;
      readBackStream.read(timestamp);
      int currentIndex = readBackStream.getIndex();
      if (currentIndex != index + 1) {
         status = PV_FAILURE;
         ErrorLog().printf(
               "testReadMultipleFrames(): "
               "after reading index %d, index was %d instead of expected %d\n",
               index, currentIndex, index + 1);
      }
      if (timestamp != timestamps[index]) {
         status = PV_FAILURE;
         ErrorLog().printf(
               "testReadMultipleFrames() index %d read timestamp %f instead of expected %f\n",
               index, timestamp, timestamps[index]);
      }
      if (compareLayerData(testData[index], readBackData) != PV_SUCCESS) {
         status = PV_FAILURE;
         ErrorLog().printf("testReadMultipleFrames() failed on index %d.\n", index);
      }
   }
   return status;
}

int testReadRandomAccess(
      std::shared_ptr<FileManager const> fileManager, int numFeatures, int globalBatchWidth) {
   int status = PV_SUCCESS;
   auto mpiBlock = fileManager->getMPIBlock();
   std::string filename("testReadRandomAccess.pvp");
   std::vector<double> timestamps{28.0, 30.0, 32.0, 34.0};
   std::vector<int> starts{14, 15, 16, 17};
   int step = 3;
   int batchStep = 16;

   // Make test data using FileStream primitive functions, without using SparseBroadcastLayerFile.
   std::vector<std::vector<SparseList<float>>> testData(4);
   for (int index = 0; index < 4; ++index) {
      testData[index] = makeSparseBroadcastLayerData(
            mpiBlock, numFeatures, globalBatchWidth, starts[index], step, batchStep);
      writeUsingFileStreamPrimitives(
            fileManager, filename, testData[index], timestamps[index], index);
   }

   // Create SparseBroadcastLayerFile object to read the data back.
   int localBatchWidth = globalBatchWidth / mpiBlock->getGlobalBatchDimension();
   SparseBroadcastLayerFile readBackStream(
         fileManager,
         filename,
         numFeatures,
         localBatchWidth,
         true /*readOnlyFlag*/,
         false /*clobberFlag*/,
         false /*verifyWrites*/);
   std::vector<SparseList<float>> readBackData(localBatchWidth);
   for (int b = 0; b < localBatchWidth; ++b) {
      readBackData[b].reset(1, 1, numFeatures);
      readBackStream.setListLocation(&readBackData[b], b);
   }

   if (status == PV_SUCCESS) {
      int startingIndex = readBackStream.getIndex();
      if (startingIndex != 0) {
         ErrorLog().printf(
               "testReadRandomAccess() expected index to initialize with value 0; "
               "instead it was %d\n",
               startingIndex);
         status = PV_FAILURE;
      }
   }

   // Position the stream to index 1.
   if (status == PV_SUCCESS) {
      readBackStream.setIndex(1);
      for (int b = 0; b < localBatchWidth; ++b) {
         std::vector<SparseList<float>> readIndex1(1);
         readBackData[b].reset(1, 1, numFeatures);
         readBackStream.setListLocation(&readBackData[b], b);
      }
      double timestamp;
      readBackStream.read(timestamp);
      if (timestamp != timestamps[1]) {
         status = PV_FAILURE;
         ErrorLog().printf(
               "testRandomAccess() timestamps do not match: expected %f, observed %f\n",
               timestamps[1], timestamp);
      }
      if (compareLayerData(testData[1], readBackData) != PV_SUCCESS) {
         status = PV_FAILURE;
         ErrorLog().printf("testRandomAccess() failed random access read of index 1.\n");
      }
   }
   if (status != PV_SUCCESS) {
      ErrorLog().printf("testRandomAccess() failed.\n");
   }
   return status;
}

int testWrite(
      std::shared_ptr<FileManager const> fileManager, int numFeatures, int globalBatchWidth) {
   int status = PV_SUCCESS;
   int start      = 2;
   int step       = 3;
   int batchStep  = 16;
   double timestamp = 20.0;

   // Create a test file using SparseBroadcastLayerFile
   std::string filename("testWrite.pvp");
   auto mpiBlock = fileManager->getMPIBlock();
   int localBatchWidth = globalBatchWidth / mpiBlock->getGlobalBatchDimension();
   SparseBroadcastLayerFile testFile(
         fileManager,
         filename,
         numFeatures,
         localBatchWidth,
         false /*readOnlyFlag*/,
         true /*clobberFlag*/,
         false /*verifyWrites*/);
   auto layerData = makeSparseBroadcastLayerData(
         mpiBlock, numFeatures, globalBatchWidth, start, step, batchStep);
   for (int b = 0; b < localBatchWidth; ++b) {
      testFile.setListLocation(&layerData[b], b);
   }
   testFile.write(timestamp);

   // Read back the data using FileStream, without using SparseBroadcastLayerFile, and compare
   std::vector<SparseList<float>> dataFromFile(localBatchWidth);
   status = readUsingFileStreamPrimitives(
         fileManager,
         filename,
         dataFromFile,
         nullptr /*timestampPtr*/,
         0 /*index*/);
   if (status == PV_SUCCESS) {
      status = compareLayerData(layerData, dataFromFile);
   }
   if (status != PV_SUCCESS) {
      ErrorLog().printf("testWrite() failed.\n");
   }
   return status;
}

int testWriteMultipleFrames(
      std::shared_ptr<FileManager const> fileManager, int numFeatures, int globalBatchWidth) {
   int status = PV_SUCCESS;
   std::string filename("testWriteMultipleFrames.pvp");
   auto mpiBlock = fileManager->getMPIBlock();
   int localBatchWidth = globalBatchWidth / mpiBlock->getGlobalBatchDimension();
   std::vector<int> starts{10, 15, 20, 25};
   int step        = 3;
   int batchStep   = 16;
   std::vector<double> timestamps{21.0, 22.0, 23.0, 24.0};

   // Create a test file using SparseBroadcastLayerFile
   SparseBroadcastLayerFile testFile(
         fileManager,
         filename,
         numFeatures,
         localBatchWidth,
         false /*readOnlyFlag*/,
         true /*clobberFlag*/,
         false /*verifyWrites*/);
   for (int index = 0; index < 4; ++index) {
      auto layerData = makeSparseBroadcastLayerData(
            mpiBlock, numFeatures, globalBatchWidth, starts[index], step, batchStep);
      for (int b = 0; b < localBatchWidth; ++b) {
         testFile.setListLocation(&layerData[b], b);
      }
      testFile.write(timestamps[index]);
      int currentIndex = testFile.getIndex();
      if (currentIndex != index + 1) {
         status = PV_FAILURE;
         ErrorLog().printf(
               "testWriteMultipleFrames(): "
               "after writing index %d, index was %d instead of expected %d\n",
               index, currentIndex, index + 1);
      }
   }

   // Read back the data using FileStream, without using SparseBroadcastLayerFile, and compare
   std::vector<SparseList<float>> dataFromFile(localBatchWidth);
   for (int index = 0; index < 4; ++index) {
      double timestamp;
      status = readUsingFileStreamPrimitives(
            fileManager,
            filename,
            dataFromFile,
            &timestamp,
            index);
      if (timestamp != timestamps[index]) {
         status = PV_FAILURE;
         ErrorLog().printf(
               "testWriteMultipleFrames() index %d read timestamp %f instead of expected %f\n",
               index, timestamp, timestamps[index]);
      }
      if (status == PV_SUCCESS) {
         auto layerData = makeSparseBroadcastLayerData(
               mpiBlock, numFeatures, globalBatchWidth, starts[index], step, batchStep);
         status = compareLayerData(layerData, dataFromFile);
      }
      if (status != PV_SUCCESS) {
         ErrorLog().printf("testWrite() failed.\n");
      }
   }

   // Test random-access
   if (status == PV_SUCCESS) {
      starts[1] = 7;
      timestamps[1] = 100.0;
      testFile.setIndex(1);
      auto layerData = makeSparseBroadcastLayerData(
            mpiBlock, numFeatures, globalBatchWidth, starts[1], step, batchStep);
      for (int b = 0; b < localBatchWidth; ++b) {
         testFile.setListLocation(&layerData[b], b);
      }
      testFile.write(timestamps[1]);
      testFile.setIndex(0);
      for (int index = 0; index <= 1; ++index) {
         double timestamp;
         status = readUsingFileStreamPrimitives(
               fileManager,
               filename,
               dataFromFile,
               &timestamp,
               index);
         if (timestamp != timestamps[index]) {
            status = PV_FAILURE;
            ErrorLog().printf(
                  "testWriteMultipleFrames() index %d read timestamp %f instead of expected %f\n",
                  index, timestamp, timestamps[index]);
         }
         if (status == PV_SUCCESS) {
            auto layerData = makeSparseBroadcastLayerData(
                  mpiBlock, numFeatures, globalBatchWidth, starts[index], step, batchStep);
            status = compareLayerData(layerData, dataFromFile);
         }
         if (status != PV_SUCCESS) { break; }
      }
   }
   if (status != PV_SUCCESS) {
      ErrorLog().printf("testWrite() failed.\n");
   }

   return status;
}

int testTruncate(
      std::shared_ptr<FileManager const> fileManager, int numFeatures, int globalBatchWidth) {
   int status = PV_SUCCESS;
   std::string filename("testTruncate.pvp");
   auto mpiBlock = fileManager->getMPIBlock();
   int localBatchWidth = globalBatchWidth / mpiBlock->getGlobalBatchDimension();
   std::vector<int> starts{10, 15, 20, 25};
   int step        = 3;
   int batchStep   = 12;
   std::vector<double> timestamps{25.0, 26.0, 27.0, 28.0};

   // Create a test file with four frames.
   SparseBroadcastLayerFile testFile(
         fileManager,
         filename,
         numFeatures,
         localBatchWidth,
         false /*readOnlyFlag*/,
         true /*clobberFlag*/,
         false /*verifyWrites*/);
   for (int index = 0; index < 4; ++index) {
      auto layerData = makeSparseBroadcastLayerData(
            mpiBlock, numFeatures, globalBatchWidth, starts[index], step, batchStep);
      for (int b = 0; b < localBatchWidth; ++b) {
         testFile.setListLocation(&layerData[b], b);
      }
      testFile.write(timestamps[index]);
      int currentIndex = testFile.getIndex();
      if (currentIndex != index + 1) {
         status = PV_FAILURE;
         ErrorLog().printf(
               "testWriteMultipleFrames(): "
               "after writing index %d, index was %d instead of expected %d\n",
               index, currentIndex, index + 1);
      }
   }

   // Truncate file to two frames
   testFile.truncate(2);

   return status;
}

int writeUsingFileStreamPrimitives(
      std::shared_ptr<FileManager const> fileManager,
      std::string const &path,
      std::vector<SparseList<float>> sparseLayerData,
      double timestamp,
      int index) {
   int status = PV_SUCCESS;
   auto fileStream = FileStreamBuilder(
      fileManager,
      path,
      false /*isTextFlag*/,
      false /*readOnlyFlag*/,
      false /*clobberFlag*/,
      false /*verifyWritesFlag*/).get();
   auto mpiBlock = fileManager->getMPIBlock();
   int numBatchProcs = mpiBlock->getBatchDimension();
   int localBatchWidth = static_cast<int>(sparseLayerData.size());
   int fileBatchWidth = localBatchWidth * numBatchProcs;
   int numFeatures = static_cast<int>(sparseLayerData[0].getFeatures());
   if (fileStream) {
      fileStream->setInPos(0L, std::ios_base::end);
      long int fileSize = fileStream->getInPos();
      BufferUtils::SparseFileTable sparseFileTable;
      if (fileSize > 0L) {
         pvAssert(fileSize > 80L);
         fileStream->setInPos(0L, std::ios_base::beg);
         BufferUtils::ActivityHeader fileHeader;
         fileStream->read(&fileHeader, sizeof(fileHeader));
         int numBands = fileHeader.nBands;
         sparseFileTable = BufferUtils::buildSparseFileTable(*fileStream, numBands - 1);
      }
      int numFrames = static_cast<int>(sparseFileTable.frameStartOffsets.size());
      int newNumFrames = std::max(numFrames, fileBatchWidth * (index + 1));
      auto header =
            BufferUtils::buildSparseActivityHeader<float>(1, 1, numFeatures, newNumFrames);
      fileStream->setInPos(0L, std::ios_base::beg);
      fileStream->setOutPos(0L, std::ios_base::beg);
      fileStream->write(&header, 80L);
      long int newFilePos = 80L;
      if (sparseFileTable.frameStartOffsets.empty() == false) {
         int bandIndex = index * fileBatchWidth - 1;
         newFilePos = sparseFileTable.frameStartOffsets.at(bandIndex);
         newFilePos += sparseFileTable.frameLengths.at(bandIndex) * static_cast<long>(sizeof(long));
         newFilePos += static_cast<long>(sizeof(double) + sizeof(int));
      }
      fileStream->setInPos(newFilePos, std::ios_base::beg);
      fileStream->setOutPos(newFilePos, std::ios_base::beg);

      std::vector<std::vector<uint32_t>> gatheredDataIndices(fileBatchWidth);
      std::vector<std::vector<float>> gatheredDataValues(fileBatchWidth);
      for (int m = 0; m < numBatchProcs; ++m) {
         for (int b = 0; b < localBatchWidth; ++b) {
            if (m == 0) {
               auto dataEntries = sparseLayerData.at(b).getContents();
               auto numEntries = dataEntries.size();
               gatheredDataIndices.at(b).resize(numEntries);
               gatheredDataValues.at(b).resize(numEntries);
               for (decltype(numEntries) n = 0; n < numEntries; ++n) {
                  gatheredDataIndices.at(b)[n] = dataEntries[n].index;
                  gatheredDataValues.at(b)[n] = dataEntries[n].value;
               }
            }
            else {
               int fileBatchIndex = b + localBatchWidth * m;
               int rank = mpiBlock->calcRankFromRowColBatch(0, 0, m);
               MPI_Status mpiStatus;
               MPI_Probe(rank, 333 + b /*tag*/, mpiBlock->getComm(), &mpiStatus);
               int count;
               MPI_Get_count(&mpiStatus, MPI_UNSIGNED, &count);
               gatheredDataIndices.at(fileBatchIndex).resize(count);
               MPI_Recv(
                     gatheredDataIndices.at(fileBatchIndex).data(),
                     count,
                     MPI_UNSIGNED,
                     rank,
                     333 + b /*tag*/,
                     mpiBlock->getComm(),
                     MPI_STATUS_IGNORE);
               gatheredDataValues.at(fileBatchIndex).resize(count);
               MPI_Recv(
                     gatheredDataValues.at(fileBatchIndex).data(),
                     count,
                     MPI_UNSIGNED,
                     rank,
                     1333 + b /*tag*/,
                     mpiBlock->getComm(),
                     MPI_STATUS_IGNORE);
            }
         } // loop over b ( local batch elements)
      } // loop over m (batch processes)
      for (int fileBatchIndex = 0; fileBatchIndex < fileBatchWidth; ++fileBatchIndex) {
         fileStream->write(&timestamp, static_cast<long>(sizeof(double)));
         int count = static_cast<int>(gatheredDataIndices.at(fileBatchIndex).size());
         fileStream->write(&count, static_cast<long>(sizeof(int)));
         std::vector<SparseList<float>::Entry> gatheredDataEntries(count);
         for (int n = 0; n < count; ++n) {
            gatheredDataEntries.at(n).index = gatheredDataIndices.at(fileBatchIndex).at(n);
            gatheredDataEntries.at(n).value = gatheredDataValues.at(fileBatchIndex).at(n);
         }

         long int sizeInBytes = count * static_cast<long>(sizeof(SparseList<float>::Entry));
         fileStream->write(gatheredDataEntries.data(), sizeInBytes);
      }
   } // end of if-clause for root process
   else { // nonroot process
      // Broadcast, so only one (row,column) pair needs to send
      if (mpiBlock->getRowIndex() == 0 and mpiBlock->getColumnIndex() == 0) {
         for (int b = 0; b < localBatchWidth; ++b) {
            auto dataEntries = sparseLayerData.at(b).getContents();
            auto numEntries = dataEntries.size();
            std::vector<uint32_t> dataIndices(numEntries);
            std::vector<float> dataValues(numEntries);
            for (decltype(numEntries) n = 0; n < numEntries; ++n) {
               dataIndices[n] = dataEntries[n].index;
               dataValues[n] = dataEntries[n].value;
            }
            MPI_Send(
                  dataIndices.data(),
                  (int)numEntries,
                  MPI_UNSIGNED,
                  0 /*dest rank*/,
                  333 + b /*tag*/,
                  mpiBlock->getComm());
            MPI_Send(
                  dataValues.data(),
                  (int)numEntries,
                  MPI_UNSIGNED,
                  0 /*dest rank*/,
                  1333 + b /*tag*/,
                  mpiBlock->getComm());
         }
      }
   } // end of else-clause for root process
   return status;
}
