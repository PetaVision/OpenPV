/*
 * BroadcastLayerFileTest.cpp
 *
 */

#include "columns/PV_Init.hpp"
#include "io/BroadcastLayerFile.hpp"
#include "io/FileManager.hpp"
#include "io/FileStreamBuilder.hpp"
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

// Calculates the file position of the start of a given index.
// (If the file has n batch elements, as specified by the mpiBlock and the localBatchWidth
// arguments, index k corresponds to PVP-file frame k*n.)
long int calcFilePosition(
      std::shared_ptr<MPIBlock const> mpiBlock, int localBatchWidth, int numFeatures, int index);

// Calculates the size of one PVP-file frame, in bytes.
long int calcFrameSize(int numFeatures);

// Recursively deletes the contents of the directory specified by path, and removes the directory
// itself, unless path is "." or ends in "/."
int cleanDirectory(std::shared_ptr<FileManager const> fileManager, std::string const &path);

int compareLayerData(
      std::vector<std::vector<float>> const &expected,
      std::vector<std::vector<float>> const &observed);

std::vector<std::vector<float>> makeBroadcastLayerData(
      std::shared_ptr<MPIBlock const> mpiBlock,
      int numFeatures,
      int globalBatchWidth,
      float start,
      float step,
      float batchStep);

// Read data from a layer .pvp file using only FileStream methods.
// On entry, layerData is a vector of the expected localBatchWidth
// (because of M-to-N, this may not the the batch width of the file).
// If timestampPtr is null, the timestamp of the file is ignored.
// If it is not null, the timestamp is filled with the file's timestamp
// for the specified index.
// The index argument is used the same way it is used in the
// BroadcastLayerFile's getIndex() and setIndex() function members.
int readUsingFileStreamPrimitives(
      std::shared_ptr<FileManager const> fileManager,
      std::string const &path,
      std::vector<std::vector<float>> &layerData,
      double *timestampPtr,
      int index);

int runTests(std::shared_ptr<FileManager> fileManager);

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
      std::vector<std::vector<float>> layerData,
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

long int calcFilePosition(
      std::shared_ptr<MPIBlock const> mpiBlock, int localBatchWidth, int numFeatures, int index) {
   long int frameSize = calcFrameSize(numFeatures);
   long int framesPerIndex = static_cast<long int>(localBatchWidth * mpiBlock->getBatchDimension());
   long int indexSize = frameSize * framesPerIndex;
   long int filePosition = 80L + indexSize * static_cast<long int>(index); // header is 80 bytes
   return filePosition;
}

long int calcFrameSize(int numFeatures) {
   long int frameDataSize =
         static_cast<long int>(numFeatures) * static_cast<long int>(sizeof(float));
   long int frameSize = static_cast<long int>(sizeof(double)) + frameDataSize; // Add timestamp
   return frameSize;
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
      std::vector<std::vector<float>> const &expected,
      std::vector<std::vector<float>> const &observed) {
   if (expected.size() != observed.size()) {
      ErrorLog() << "compareLayerData() called with expected size " << expected.size()
                 << " and observed size " << observed.size() << "\n";
      return PV_FAILURE;
   }
   int status = PV_SUCCESS;
   int batchWidth = static_cast<int>(expected.size());
   for (int b = 0; b < batchWidth; ++b) {
      auto const &expectedElem = expected[b];
      auto const &observedElem = observed[b];
      int numFeatures = (int)expectedElem.size();
      if (observedElem.size() != numFeatures) {
         ErrorLog() << "compareLayerData() expected batch element " << b
                    << " has size " << numFeatures << " but observed batch element "
                    << " has size " << observedElem.size() << "\n";
         status = PV_FAILURE;
      }
      else {
         for (int f = 0; f < numFeatures; ++f) {
            float observedValue = observed[b][f];
            float expectedValue = expected[b][f];
            if (observedValue != expectedValue) {
               ErrorLog().printf(
                     "compareLayerData() batch element %d, feature %d: expected %f, observed %f"
                     " (discrepancy %g)\n",
                     b, f, static_cast<double>(expectedValue), static_cast<double>(observedValue),
                     static_cast<double>(observedValue) - static_cast<double>(expectedValue));
               status = PV_FAILURE;
            }
         }
      }
   }
   return status;
}

std::vector<std::vector<float>> makeBroadcastLayerData(
      std::shared_ptr<MPIBlock const> mpiBlock,
      int numFeatures,
      int globalBatchWidth,
      float start,
      float step,
      float batchStep) {
   int status = PV_SUCCESS;
   int localBatchWidth = globalBatchWidth / mpiBlock->getGlobalBatchDimension();
   pvAssert(localBatchWidth * mpiBlock->getGlobalBatchDimension() == globalBatchWidth);
   int kb0 = localBatchWidth * (mpiBlock->getStartBatch() + mpiBlock->getBatchIndex());
   std::vector<std::vector<float>> result(localBatchWidth);
   for (int b = 0; b < localBatchWidth; ++b) {
      int globalBatchIndex = b + kb0;
      float batchOffset = batchStep * static_cast<float>(globalBatchIndex);
      result[b].resize(numFeatures);
      for (int f = 0; f < numFeatures; ++f) {
         result[b][f] = start + step * static_cast<float>(f) + batchOffset;
      }
   }
   return result;
}

int readUsingFileStreamPrimitives(
      std::shared_ptr<FileManager const> fileManager,
      std::string const &path,
      std::vector<std::vector<float>> &layerData,
      double *timestampPtr,
      int index) {
   int status = PV_SUCCESS;
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

      // Sanity check of header values against BroadcastLayerFile requirements and function args
      if (header.nx != 1 or header.ny != 1) {
         ErrorLog().printf(
               "readUsingFileStreamPrimitives(): file \"%s\" is not a 1-by-1-by-nf pvp file.\n",
               path.c_str());
         return PV_FAILURE;
      }
      int numFeatures = header.nf;
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
      long int filePos = calcFilePosition(mpiBlock, localBatchWidth, numFeatures, index);
      fileStream->setInPos(filePos, std::ios_base::beg);
      std::vector<std::vector<float>> values(expectedFileBatchWidth);
      for (int b = 0; b < expectedFileBatchWidth; ++b) {
         values[b].resize(numFeatures);
         fileStream->read(&timestamps.at(b), 8L);
         fileStream->read(values[b].data(), 4L * static_cast<long int>(numFeatures));
      }
      MPI_Bcast(
            timestamps.data(), expectedFileBatchWidth, MPI_DOUBLE, 0 /*root*/, mpiBlock->getComm());

      // Distribute values to rest of MPIBlock.
      int sendSize = localBatchWidth * numFeatures;
      std::vector<float> sendData(sendSize);
      for (int b = 0; b < localBatchWidth; ++b) {
         layerData[b].resize(numFeatures);
         std::copy(values[b].cbegin(), values[b].cend(), layerData[b].data());
      }
      for (int r = 1; r < mpiBlock->getSize(); ++r) {
         int mpiBatchIndex = mpiBlock->calcBatchIndexFromRank(r);
         for (int b = 0; b < localBatchWidth; ++b) {
            int globalBatchIndex = b + localBatchWidth * mpiBatchIndex;
            // SCRABBLE use the sendData vector to consolidate, so as to only do one MPI_Send
            MPI_Send(
                  values[globalBatchIndex].data(),
                  numFeatures,
                  MPI_FLOAT,
                  r,
                  345 + b /*tag*/,
                  mpiBlock->getComm());
         }
      }
   }
   else {
      MPI_Bcast(
            timestamps.data(), expectedFileBatchWidth, MPI_DOUBLE, 0 /*root*/, mpiBlock->getComm());
      for (int b = 0; b < localBatchWidth; ++b) {
         MPI_Status mpiStatus;
         MPI_Probe(0 /*source*/, 345 + b /*tag*/, mpiBlock->getComm(), &mpiStatus);
         int numFeatures;
         MPI_Get_count(&mpiStatus, MPI_FLOAT, &numFeatures);
         layerData[b].resize(numFeatures);
         MPI_Recv(
               layerData[b].data(),
               numFeatures,
               MPI_FLOAT,
               0 /*source*/,
               345 + b /*tag*/,
               mpiBlock->getComm(),
               MPI_STATUS_IGNORE);
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
   int numFeatures = 16;
   int globalBatchWidth = 16;
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

int testRead(
      std::shared_ptr<FileManager const> fileManager, int numFeatures, int globalBatchWidth) {
   int status = PV_SUCCESS;
   std::string filename("testRead.pvp");
   float start      = 1.0f;
   float step       = 1.0f;
   float batchStep  = 16.0f;
   double timestamp = 11.0;
   auto mpiBlock    = fileManager->getMPIBlock();

   // Create test data and write it without using BroadcastLayerFile
   auto layerData = makeBroadcastLayerData(
         mpiBlock, numFeatures, globalBatchWidth, start, step, batchStep);
   writeUsingFileStreamPrimitives(fileManager, filename, layerData, timestamp, 0);

   // Read data back using BroadcastLayerFile, and compare it to the test data
   int localBatchWidth = static_cast<int>(layerData.size());
   BroadcastLayerFile readBackStream(
         fileManager,
         filename,
         numFeatures,
         localBatchWidth,
         true /*readOnlyFlag*/,
         false /*clobberFlag*/,
         false /*verifyWrites*/);
   std::vector<std::vector<float>> readBackData(localBatchWidth);
   for (int b = 0; b < localBatchWidth; ++b) {
      readBackData[b].resize(numFeatures);
      readBackStream.setDataLocation(readBackData[b].data(), b);
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
   std::vector<float> starts{10.0f, 11.0f, 12.0f, 13.0f};
   float step = 1.0f;
   float batchStep = 16.0f;

   // Make test data using FileStream primitive functions, without using BroadcastLayerFile.
   for (int index = 0; index < 4; ++index) {
      auto layerData = makeBroadcastLayerData(
            mpiBlock, numFeatures, globalBatchWidth, starts[index], step, batchStep);
      writeUsingFileStreamPrimitives(fileManager, filename, layerData, timestamps[index], index);
   }

   // Read back the test data using BroadcastLayerFile.
   int localBatchWidth = globalBatchWidth / mpiBlock->getGlobalBatchDimension();
   BroadcastLayerFile readBackStream(
         fileManager,
         filename,
         numFeatures,
         localBatchWidth,
         true /*readOnlyFlag*/,
         false /*clobberFlag*/,
         false /*verifyWrites*/);
   std::vector<std::vector<float>> readBackData(localBatchWidth);
   for (int b = 0; b < localBatchWidth; ++b) {
      readBackData[b].resize(numFeatures);
      readBackStream.setDataLocation(readBackData[b].data(), b);
   }
   for (int index = 0; index < 4; ++index) {
      double timestamp;
      readBackStream.read(timestamp);
      int currentIndex = readBackStream.getIndex();
      if (currentIndex != index + 1) {
         status = PV_FAILURE;
         ErrorLog().printf(
               "testReadMultipleFrames(): after reading index %d, index was %d instead of expected %d\n",
               index, currentIndex, index + 1);
      }
      if (timestamp != timestamps[index]) {
         status = PV_FAILURE;
         ErrorLog().printf(
               "testReadMultipleFrames() index %d read timestamp %f instead of expected %f\n",
               index, timestamp, timestamps[index]);
      }
      auto layerData = makeBroadcastLayerData(
            mpiBlock, numFeatures, globalBatchWidth, starts[index], step, batchStep);
      if (compareLayerData(layerData, readBackData) != PV_SUCCESS) {
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
   std::vector<float> starts{14.0f, 15.0f, 16.0f, 17.0f};
   float step = 1.0f;
   float batchStep = 16.0f;

   // Make test data using FileStream primitive functions, without using BroadcastLayerFile.
   for (int index = 0; index < 4; ++index) {
      auto layerData = makeBroadcastLayerData(
            mpiBlock, numFeatures, globalBatchWidth, starts[index], step, batchStep);
      writeUsingFileStreamPrimitives(fileManager, filename, layerData, timestamps[index], index);
   }

   // Create BroadcastLayerFile object to read the data back.
   int localBatchWidth = globalBatchWidth / mpiBlock->getGlobalBatchDimension();
   BroadcastLayerFile readBackStream(
         fileManager,
         filename,
         numFeatures,
         localBatchWidth,
         true /*readOnlyFlag*/,
         false /*clobberFlag*/,
         false /*verifyWrites*/);
   std::vector<std::vector<float>> readBackData(localBatchWidth);
   for (int b = 0; b < localBatchWidth; ++b) {
      readBackData[b].resize(numFeatures);
      readBackStream.setDataLocation(readBackData[b].data(), b);
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
         std::vector<std::vector<float>> readIndex1(1);
         readBackData[b].resize(numFeatures);
         readBackStream.setDataLocation(readBackData[b].data(), b);
      }
      double timestamp;
      readBackStream.read(timestamp);
      auto layerData = makeBroadcastLayerData(
            mpiBlock, numFeatures, globalBatchWidth, starts[1], step, batchStep);
      if (timestamp != timestamps[1]) {
         status = PV_FAILURE;
         ErrorLog().printf(
               "testRandomAccess() timestamps do not match: expected %f, observed %f\n",
               timestamps[1], timestamp);
      }
      if (compareLayerData(layerData, readBackData) != PV_SUCCESS) {
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
   float start      = 2.0f;
   float step       = 1.0f;
   float batchStep  = 16.0f;
   double timestamp = 20.0;

   // Create a test file using BroadcastLayerFile
   std::string filename("testWrite.pvp");
   auto mpiBlock = fileManager->getMPIBlock();
   int localBatchWidth = globalBatchWidth / mpiBlock->getGlobalBatchDimension();
   BroadcastLayerFile testFile(
         fileManager,
         filename,
         numFeatures,
         localBatchWidth,
         false /*readOnlyFlag*/,
         true /*clobberFlag*/,
         false /*verifyWrites*/);
   auto layerData = makeBroadcastLayerData(
         mpiBlock, numFeatures, globalBatchWidth, start, step, batchStep);
   for (int b = 0; b < localBatchWidth; ++b) {
      testFile.setDataLocation(layerData[b].data(), b);
   }
   testFile.write(timestamp);

   // Read back the data using FileStream, without using BroadcastLayerFile, and compare
   std::vector<std::vector<float>> dataFromFile(localBatchWidth);
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
   std::vector<float> starts{10.0f, 15.0f, 20.0f, 25.0f};
   float step        = 1.0f;
   float batchStep   = 16.0f;
   std::vector<double> timestamps{21.0f, 22.0f, 23.0f, 24.0f};

   // Create a test file using BroadcastLayerFile
   BroadcastLayerFile testFile(
         fileManager,
         filename,
         numFeatures,
         localBatchWidth,
         false /*readOnlyFlag*/,
         true /*clobberFlag*/,
         false /*verifyWrites*/);
   for (int index = 0; index < 4; ++index) {
      auto layerData = makeBroadcastLayerData(
            mpiBlock, numFeatures, globalBatchWidth, starts[index], step, batchStep);
      for (int b = 0; b < localBatchWidth; ++b) {
         testFile.setDataLocation(layerData[b].data(), b);
      }
      testFile.write(timestamps[index]);
      int currentIndex = testFile.getIndex();
      if (currentIndex != index + 1) {
         status = PV_FAILURE;
         ErrorLog().printf(
               "testWriteMultipleFrames(): after writing index %d, index was %d instead of expected %d\n",
               index, currentIndex, index + 1);
      }
   }

   // Read back the data using FileStream, without using BroadcastLayerFile, and compare
   std::vector<std::vector<float>> dataFromFile(localBatchWidth);
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
         auto layerData = makeBroadcastLayerData(
               mpiBlock, numFeatures, globalBatchWidth, starts[index], step, batchStep);
         status = compareLayerData(layerData, dataFromFile);
      }
      if (status != PV_SUCCESS) {
         ErrorLog().printf("testWrite() failed.\n");
      }
   }

   // Test random-access
   if (status == PV_SUCCESS) {
      starts[1] = 7.0f;
      timestamps[1] = 100.0f;
      testFile.setIndex(1);
      auto layerData = makeBroadcastLayerData(
            mpiBlock, numFeatures, globalBatchWidth, starts[1], step, batchStep);
      for (int b = 0; b < localBatchWidth; ++b) {
         testFile.setDataLocation(layerData[b].data(), b);
      }
      testFile.write(timestamps[1]);
      testFile.setIndex(0);
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
            auto layerData = makeBroadcastLayerData(
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
   return status;
}

int writeUsingFileStreamPrimitives(
      std::shared_ptr<FileManager const> fileManager,
      std::string const &path,
      std::vector<std::vector<float>> layerData,
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
   int localBatchWidth = static_cast<int>(layerData.size());
   int fileBatchWidth = localBatchWidth * numBatchProcs;
   int numFeatures = static_cast<int>(layerData[0].size());
   if (fileStream) {
      fileStream->setInPos(0L, std::ios_base::end);
      long int fileSize = fileStream->getInPos();
      int numFrames = 0;
      if (fileSize > 0L) {
         pvAssert(fileSize > 80L);
         long int frameSize = calcFrameSize(numFeatures);
         numFrames = static_cast<int>((fileSize - 80L) / frameSize);
         pvAssert(fileSize == 80L + static_cast<long int>(numFrames) * frameSize);
      }
      int newNumFrames = std::max(numFrames, fileBatchWidth * (index + 1));
      auto header =
            BufferUtils::buildActivityHeader<float>(1, 1, numFeatures, newNumFrames);
      fileStream->setInPos(0L, std::ios_base::beg);
      fileStream->setOutPos(0L, std::ios_base::beg);
      fileStream->write(&header, 80L);
      long int newFilePosition =
            calcFilePosition(mpiBlock, localBatchWidth, numFeatures, index);
      fileStream->setInPos(newFilePosition, std::ios_base::beg);
      fileStream->setOutPos(newFilePosition, std::ios_base::beg);

      std::vector<float> gatheredData(fileBatchWidth * numFeatures);
      for (int b = 0; b < localBatchWidth; ++b) {
         std::copy(layerData[b].cbegin(), layerData[b].cend(), &gatheredData.at(b * numFeatures));
         for (int m = 1; m < numBatchProcs; ++m) {
            int fileBatchIndex = b + localBatchWidth * m;
            int rank = mpiBlock->calcRankFromRowColBatch(0, 0, m);
            MPI_Recv(
                  &gatheredData.at(fileBatchIndex * numFeatures),
                  numFeatures,
                  MPI_FLOAT,
                  rank,
                  333 + b /*tag*/,
                  mpiBlock->getComm(),
                  MPI_STATUS_IGNORE);
         }
      }
      for (int b = 0; b < fileBatchWidth; ++b) {
         fileStream->write(&timestamp, sizeof(timestamp));
         fileStream->write(&gatheredData.at(b * numFeatures), sizeof(float) * numFeatures);
      }
   }
   else {
      if (mpiBlock->getRowIndex() == 0 and mpiBlock->getColumnIndex() == 0) {
         for (int b = 0; b < localBatchWidth; ++b) {
            MPI_Send(
                  layerData[b].data(),
                  numFeatures,
                  MPI_FLOAT,
                  0 /*dest rank*/,
                  333 + b /*tag*/,
                  mpiBlock->getComm());
         }
      }
   }
   return status;
}
