/*
 * BroadcastPreWeightsFileTest.cpp
 */

#include <columns/Communicator.hpp>
#include <columns/PV_Init.hpp>
#include <components/LayerGeometry.hpp>
#include <include/pv_common.h>
#include <io/FileManager.hpp>
#include <io/FileStreamBuilder.hpp>
#include <io/BroadcastPreWeightsFile.hpp>
#include <structures/WeightData.hpp>
#include <utils/BufferUtilsMPI.hpp>

#include <cerrno>
#include <memory>
#include <string>
#include <vector>

using namespace PV;

struct Dimensions {
   int nfPre;
   int nxPost;
   int nyPost;
   int nfPost;
   Dimensions(Communicator const *communicator) {
      nfPre     = 8;
      nxPost    = 4 / communicator->numCommColumns();
      nyPost    = 4 / communicator->numCommRows();
      nfPost    = 3;
   }
};

int calcFrameSize(
      std::shared_ptr<WeightData const> weights, std::shared_ptr<MPIBlock const> mpiBlock);

// Creates a test file using FileStream commands, without using BroadcastPreWeightsFile; and then
// reads it back using BroadcastPreWeightsFile, and checks the contents.
int checkRead(std::shared_ptr<FileManager const> fileManager, Dimensions const &dimensions);

// Creates a multiple-frame test file using FileStream commands, without using
// BroadcastPreWeightsFile; and then reads it back using BroadcastPreWeightsFile, and checks
// the contents.
int checkReadMultiple(std::shared_ptr<FileManager const> fileManager, Dimensions const &dimensions);

// Creates a test file using BroadcastPreWeightsFile::write(), and then reads it back using
// FileStream commands, and checks the contents.
int checkWrite(std::shared_ptr<FileManager const> fileManager, Dimensions const &dimensions);

// Creates a multiple-frame test file using BroadcastPreWeightsFile::write(), and reads it
// back using both FileStream commands and BroadcastPreWeightsFile::read(), and checks the contents.
// The write and read commands are interleaved, the read and write commands may specify a frame
// other than the one most recently written.
int checkWriteMultiple(
      std::shared_ptr<FileManager const> fileManager, Dimensions const &dimensions);

// Creates a multiple-frame test file, truncates it to a smaller number of frames, and then checks
// that the size of the file is correct and that the contents of the truncated file is correct.
int checkTruncate(std::shared_ptr<FileManager const> fileManager, Dimensions const &dimensions);

// Recursively deletes the contents of the directory specified by path, and removes the directory
// itself, unless path is "." or ends in "/."
int cleanDirectory(std::shared_ptr<FileManager const> fileManager, std::string const &path);

// Checks that the weights have the same dimensions and, if so, checks that the weights have
// identical values. Returns PV_SUCCESS if the weights are the same, and PV_FAILURE otherwise.
// In case of failure, prints error messages to ErrorLog().
int compareWeights(
      std::shared_ptr<WeightData const> expected,
      std::shared_ptr<WeightData const> observed,
      std::string const &errorLabel);

// Creates a FileManager object anchored at the output directory specified in the
// PV_Init object's arguments.
std::shared_ptr<FileManager> createFileManager(std::shared_ptr<PV_Init> pv_init_obj);

// The root process of the MPIBlock gathers the weights from all the processes in the MPIBlock
// into a single WeightData structure, with dimensions
//    weightData->getPatchSizeX() * mpiBlock->getNumColumns() by
//    weightData->getPatchSizeY() * mpiBlock->getNumRows() by weightData->getpatchSizeF().
// The dimensions of the weightData object must be the same for each process.
std::shared_ptr<WeightData> gatherWeightsByBlock(
      std::shared_ptr<WeightData const> weightData,
      std::shared_ptr<MPIBlock const> mpiBlock);

// Create a WeightData object with the specified dimensions, and whose weight at value k is
// start + k*step. Used for generating weights to be used in the tests.
std::shared_ptr<WeightData> makeWeightData(
      Dimensions const &dimensions,
      std::shared_ptr<MPIBlock const> mpiBlock,
      float start,
      float step);

void setWeightDataValues(
      std::shared_ptr<WeightData> weightData,
      std::shared_ptr<MPIBlock const> mpiBlock,
      float start,
      float step);

int readUsingBroadcastPreWeightsFileAndCheck(
      BroadcastPreWeightsFile &file,
      std::shared_ptr<WeightData> weightsInFile,
      Dimensions const &dimensions,
      std::shared_ptr<MPIBlock const> mpiBlock,
      float correctStart,
      float correctStep,
      double correctTimestamp);

int readUsingFileStreamAndCheck(
      std::shared_ptr<FileManager const> fileManager,
      std::string const &path,
      Dimensions const &dimensions,
      std::shared_ptr<MPIBlock const> mpiBlock,
      float correctStart,
      float correctStep,
      double correctTimestamp,
      int frameNumber);

std::shared_ptr<WeightData> readUsingFileStreamPrimitives(
      std::shared_ptr<FileManager const> fileManager,
      std::string const &path,
      Dimensions const &dimensions,
      int frameNumber,
      double &timestamp);

void scatterWeightsByBlock(
      std::shared_ptr<WeightData> weightData,
      std::shared_ptr<WeightData const> gatheredWeightData,
      std::shared_ptr<MPIBlock const> mpiBlock);

int verifyFrameIndex(BroadcastPreWeightsFile &file, int correctFrame, std::string filename);

// Creates a file, using FileStream methods, containing the indicated weight data and timestamp.
// Used to create data used in testing the BroadcastPreWeightsFile::read() function member.
void writeUsingFileStreamPrimitives(
      std::shared_ptr<FileManager const> fileManager,
      std::string const &path,
      std::shared_ptr<WeightData const> weightData,
      double timestamp,
      int frameNumber);

int main(int argc, char *argv[]) {
   int status = PV_SUCCESS;

   auto pv_init = std::make_shared<PV_Init>(&argc, &argv, false /*allowUnrecognizedArgumentsFlag*/);
   std::shared_ptr<FileManager> fileManager = createFileManager(pv_init);

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

   Dimensions dimensions(pv_init->getCommunicator());
   if (status == PV_SUCCESS) {
      status = checkRead(fileManager, dimensions);
   }
   if (status == PV_SUCCESS) {
      status = checkReadMultiple(fileManager, dimensions);
   }
   if (status == PV_SUCCESS) {
      status = checkWrite(fileManager, dimensions);
   }
   if (status == PV_SUCCESS) {
      status = checkWriteMultiple(fileManager, dimensions);
   }
   if (status == PV_SUCCESS) {
      status = checkTruncate(fileManager, dimensions);
   }
   return status;
}

int calcFrameSize(
      std::shared_ptr<WeightData const> weights, std::shared_ptr<MPIBlock const> mpiBlock) {
   int numPatches = static_cast<int>(weights->getNumDataPatchesOverall());
   int nxpBlock   = weights->getPatchSizeX();
   int nypBlock   = weights->getPatchSizeY();
   if (mpiBlock != nullptr) {
      nxpBlock *= mpiBlock->getNumColumns();
      nypBlock *= mpiBlock->getNumRows();
   }
   int nfp = weights->getPatchSizeF();
   return 104 + numPatches * (8 + nxpBlock * nypBlock * nfp * 4);
}

int checkRead(std::shared_ptr<FileManager const> fileManager, Dimensions const &dimensions) {
   // Create test data
   auto mpiBlock = fileManager->getMPIBlock();
   auto weightData = makeWeightData(dimensions, mpiBlock, 0.0f /*start*/, 1.0f /*step*/);

   // Create test file
   std::string filename("checkRead_W.pvp");
   if (fileManager->queryFileExists(filename)) { fileManager->deleteFile(filename); }
   writeUsingFileStreamPrimitives(
         fileManager, filename, weightData, 11.0 /*timestamp*/, 0 /*frameNumber*/);

   // Read the weights back into a WeightData object. We initialize with wrong values so that the
   // only way for the test to pass is by reading in the correct values.
   auto readWeightData = makeWeightData(dimensions, mpiBlock, 100.0f /*start*/, -0.25f /*step*/);
   BroadcastPreWeightsFile testRead(
         fileManager,
         filename,
         readWeightData,
         dimensions.nfPre,
         false /*compressedFlag*/,
         true /*readOnlyFlag*/,
         false /*clobberFlag*/,
         false /*verifyWrites*/);
   testRead.read();

   // Compare weights we read in to the weights we wrote
   int status = compareWeights(weightData, readWeightData, "checkRead()");
   if (status != PV_SUCCESS) {
      ErrorLog().printf("checkRead failed for \"%s\"\n", filename.c_str());
   }
   return status;
}

int checkReadMultiple(
      std::shared_ptr<FileManager const> fileManager, Dimensions const &dimensions) {
   auto mpiBlock = fileManager->getMPIBlock();
   std::string filename("checkReadMultiple_W.pvp");
   std::vector<double> timestamps{20.0, 22.0, 24.0, 26.0};
   std::vector<float> starts{10.0f, 11.0f, 12.0f, 13.0f};
   for (int frame = 0; frame < 4; ++frame) {
      auto weightData = makeWeightData(dimensions, mpiBlock, starts.at(frame), 1.0f /*step*/);
      writeUsingFileStreamPrimitives(
            fileManager, filename, weightData, timestamps.at(frame), frame);
   }

   // Read the weights back into a WeightData object. We initialize with wrong values so that the
   // only way for the test to pass is by reading in the correct values.
   auto readWeightData = makeWeightData(dimensions, mpiBlock, 100.0f /*start*/, -0.25f /*step*/);
   BroadcastPreWeightsFile testRead(
         fileManager,
         filename,
         readWeightData,
         dimensions.nfPre,
         false /*compressedFlag*/,
         true /*readOnlyFlag*/,
         false /*clobberFlag*/,
         false /*verifyWrites*/);
   FatalIf(
         testRead.getFileStream() and testRead.getNumFrames() != 4,
         "checkReadMultiple expected \"%s\" to have 4 frames but it has %d\n",
         filename.c_str(),
         testRead.getNumFrames());
   int status = PV_SUCCESS;
   for (int frame = 0; frame < 4; ++frame) {
      int frameStatus = readUsingBroadcastPreWeightsFileAndCheck(
            testRead,
            readWeightData,
            dimensions,
            mpiBlock,
            starts.at(frame),
            1.0f,
            timestamps.at(frame));
      if (frameStatus != PV_SUCCESS) {
         ErrorLog().printf("Error reading frame %d of \"%s\"\n", frame, filename.c_str());
         status = PV_FAILURE;
      }
   }

   // Test random access of a frame
   int frame = 2;
   if (status == PV_SUCCESS) {
      testRead.setIndex(frame);
      int retrievedIndex = testRead.getIndex();
      FatalIf(
            retrievedIndex != frame,
            "After setting frame index of \"%s\" to %d, getIndex() returned %d\n",
            filename.c_str(), frame, retrievedIndex);
      double timestamp;
      auto correctWeightData = makeWeightData(
            dimensions, mpiBlock, starts.at(frame), 1.0f /*step*/);
      status = readUsingBroadcastPreWeightsFileAndCheck(
            testRead,
            readWeightData,
            dimensions,
            mpiBlock,
            starts.at(frame),
            1.0f,
            timestamps.at(frame));
      if (status != PV_SUCCESS) {
         ErrorLog().printf(
               "Error in random access reading of frame %d of \"%s\"\n", 
               frame, filename.c_str());
         status = PV_FAILURE;
      }
   }
   return status;
}

int checkWrite(std::shared_ptr<FileManager const> fileManager, Dimensions const &dimensions) {
   std::string filename("checkWrite_W.pvp");
   auto writeData = std::make_shared<WeightData>(
         1 /*numArbors*/,
         dimensions.nxPost, dimensions.nyPost, dimensions.nfPost,
         1 /*numDataPatchesX*/, 1 /*numDataPatchesY*/, dimensions.nfPre);
   BroadcastPreWeightsFile testWrite(
         fileManager,
         filename,
         writeData,
         dimensions.nfPre,
         false /*compressedFlag*/,
         false /*readOnlyFlag*/,
         false /*clobberFlag*/,
         false /*verifyWrites*/);
   auto mpiBlock = fileManager->getMPIBlock();
   float start = 100.0f;
   float step = 1.0f;
   double timestamp = 2.0;
   setWeightDataValues(writeData, mpiBlock, start, step);
   testWrite.write(timestamp);

   int status = readUsingFileStreamAndCheck(
         fileManager, filename, dimensions, mpiBlock, start, step, timestamp, 0);
   return status;
}

int checkWriteMultiple(
      std::shared_ptr<FileManager const> fileManager, Dimensions const &dimensions) {
   int status = PV_SUCCESS;
   std::string filename("checkWriteMultiple_W.pvp");
   auto writeData = std::make_shared<WeightData>(
         1 /*numArbors*/,
         dimensions.nxPost, dimensions.nyPost, dimensions.nfPost,
         1 /*numDataPatchesX*/, 1 /*numDataPatchesY*/, dimensions.nfPre);
   BroadcastPreWeightsFile testWrite(
         fileManager,
         filename,
         writeData,
         dimensions.nfPre,
         false /*compressedFlag*/,
         false /*readOnlyFlag*/,
         false /*clobberFlag*/,
         false /*verifyWrites*/);
   auto mpiBlock = fileManager->getMPIBlock();
   std::vector<float> starts{40.0f, 50.0f, 24.f};
   std::vector<float> steps{1.0f, 1.0f, 0.5f};
   std::vector<double> timestamps{2.0, 2.5, 2.75};

   if (status == PV_SUCCESS) {
      setWeightDataValues(writeData, mpiBlock, starts[0], steps[0]);
      testWrite.write(timestamps[0]);
      status = verifyFrameIndex(testWrite, 1, filename);
   }
   if (status == PV_SUCCESS) {
      status = readUsingFileStreamAndCheck(
            fileManager, filename, dimensions, mpiBlock, starts[0], steps[0], timestamps[0], 0);
      if (status != PV_SUCCESS) {
         ErrorLog().printf("checkWriteMultiple() failed for frame %d.\n", 0);
         status = PV_FAILURE;
      }
   }
   if (status == PV_SUCCESS) {
      setWeightDataValues(writeData, mpiBlock, starts[1], steps[1]);
      testWrite.write(timestamps[1]);
      status = verifyFrameIndex(testWrite, 2, filename);
   }
   if (status == PV_SUCCESS) {
      status = readUsingFileStreamAndCheck(
            fileManager, filename, dimensions, mpiBlock, starts[0], steps[0], timestamps[0], 0);
      if (status != PV_SUCCESS) {
         ErrorLog().printf("checkWriteMultiple() failed for frame %d.\n", 0);
         status = PV_FAILURE;
      }
   }
   if (status == PV_SUCCESS) {
      status = readUsingFileStreamAndCheck(
            fileManager, filename, dimensions, mpiBlock, starts[1], steps[1], timestamps[1], 1);
      if (status != PV_SUCCESS) {
         ErrorLog().printf("checkWriteMultiple() failed for frame %d.\n", 1);
         status = PV_FAILURE;
      }
   }
   if (status == PV_SUCCESS) {
      setWeightDataValues(writeData, mpiBlock, starts[2], steps[2]);
      testWrite.write(timestamps[2]);
      status = verifyFrameIndex(testWrite, 3, filename);
   }
   if (status == PV_SUCCESS) {
      status = readUsingFileStreamAndCheck(
            fileManager, filename, dimensions, mpiBlock, starts[2], steps[2], timestamps[2], 2);
      if (status != PV_SUCCESS) {
         ErrorLog().printf("checkWriteMultiple() failed for frame %d.\n", 2);
         status = PV_FAILURE;
      }
   }

   // Overwrite the first frame; make sure all three frames are correct.
   starts[0] = 1.0f;
   steps[0] =  2.0f;
   timestamps[0] = 10.0f;
   if (status == PV_SUCCESS) {
      testWrite.setIndex(0);
      verifyFrameIndex(testWrite, 0, filename);
   }
   if (status == PV_SUCCESS) {
      setWeightDataValues(writeData, mpiBlock, starts[0], steps[0]);
      testWrite.write(timestamps[0]);
      status = verifyFrameIndex(testWrite, 1, filename);
   }
   for (int f = 0; f < 3; ++f) {
      if (status == PV_SUCCESS) {
         status = readUsingFileStreamAndCheck(
               fileManager, filename, dimensions, mpiBlock, starts[f], steps[f], timestamps[f], f);
         if (status != PV_SUCCESS) {
            ErrorLog().printf("checkWriteMultiple() failed for frame %d.\n", f);
            status = PV_FAILURE;
         }
      }
   }
   return status;
}

int checkTruncate(std::shared_ptr<FileManager const> fileManager, Dimensions const &dimensions) {
   int status = PV_SUCCESS;
   std::string filename("checkTruncate_W.pvp");
   auto writeData = std::make_shared<WeightData>(
         1 /*numArbors*/,
         dimensions.nxPost, dimensions.nyPost, dimensions.nfPost,
         1 /*numDataPatchesX*/, 1 /*numDataPatchesY*/, dimensions.nfPre);
   BroadcastPreWeightsFile testTrunc(
         fileManager,
         filename,
         writeData,
         dimensions.nfPre,
         false /*compressedFlag*/,
         false /*readOnlyFlag*/,
         false /*clobberFlag*/,
         false /*verifyWrites*/);
   auto mpiBlock = fileManager->getMPIBlock();
   std::vector<float> starts{10.0f, 20.0f, 30.f, 40.0f, 50.f};
   std::vector<float> steps{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
   std::vector<double> timestamps{5.5, 10.5, 15.5, 20.5, 25.5};

   if (status == PV_SUCCESS) {
      for (int frame = 0; frame < 5; ++frame) {
         setWeightDataValues(writeData, mpiBlock, starts[frame], steps[frame]);
         testTrunc.write(timestamps[frame]);
      }
   }
   if (status == PV_SUCCESS) {
      status = verifyFrameIndex(testTrunc, 5, filename);
   }
   if (status == PV_SUCCESS) {
      testTrunc.truncate(3);
      int numFrames = testTrunc.getNumFrames();
      if (fileManager->isRoot() and numFrames != 3) {
         ErrorLog().printf(
               "After truncating, \"%s\" had %d frames instead of the expected %d\n",
               filename.c_str(), numFrames, 3);
         status = PV_FAILURE;
      }
   }
   for (int f = 0; f < 3; ++f) {
      if (status == PV_SUCCESS) {
         status = readUsingFileStreamAndCheck(
               fileManager, filename, dimensions, mpiBlock, starts[f], steps[f], timestamps[f], f);
         if (status != PV_SUCCESS) {
            ErrorLog().printf("After truncating, checkTruncate() failed for frame %d.\n", f);
            status = PV_FAILURE;
         }
      }
   }
   return status;
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
      std::shared_ptr<WeightData const> expected,
      std::shared_ptr<WeightData const> observed,
      std::string const &errorLabel) {
   int status = PV_SUCCESS;
   FatalIf(
         expected->getNumArbors() != 1,
         "compareWeights() called with expected weights having multiple arbors.\n");
   FatalIf(
         observed->getNumArbors() != 1,
         "compareWeights() called with observed weights having multiple arbors.\n");
   long int numValues = expected->getNumValuesPerArbor();
   if (observed->getNumValuesPerArbor() != numValues) {
      ErrorLog().printf(
            "%s: observed number of weight values %ld differs from expected number %ld\n",
            errorLabel.c_str(), numValues, observed->getNumValuesPerArbor());
      return PV_FAILURE;
   }

   for (long int k = 0; k < numValues; ++k) {
      float expectedValue = expected->getData(0 /*arbor*/)[k];
      float observedValue = observed->getData(0 /*arbor*/)[k];
      if (observedValue != expectedValue) {
         ErrorLog().printf("%s, weight %ld: expected %f, observed %f, discrepancy %g\n",
               errorLabel.c_str(),
               k,
               static_cast<double>(expectedValue),
               static_cast<double>(observedValue),
               static_cast<double>(observedValue - expectedValue));
         status = PV_FAILURE;
      }
   }
   return status;
}

int compareWeightsAndTimestamps(
      double expectedTimestamp,
      double observedTimestamp,
      std::shared_ptr<WeightData const> expectedWeights,
      std::shared_ptr<WeightData const> observedWeights,
      std::string const &errorLabel) {
   int status = PV_SUCCESS;
   if (observedTimestamp != expectedTimestamp) {
      ErrorLog().printf(
            "%s timestamp error: expected %f, observed %f\n",
            errorLabel.c_str(), expectedTimestamp, observedTimestamp);
      status = PV_FAILURE;
   }
   int weightCompare =
         compareWeights(expectedWeights, observedWeights, errorLabel);
   if (weightCompare != PV_SUCCESS) {
      ErrorLog().printf("%s weights do not match.\n", errorLabel.c_str());
      status = PV_FAILURE;
   }
   return status;
}

std::shared_ptr<FileManager> createFileManager(std::shared_ptr<PV_Init> pv_init_obj) {
   auto mpiBlock  = pv_init_obj->getCommunicator()->getIOMPIBlock();
   auto arguments = pv_init_obj->getArguments();
   std::string baseDirectory = arguments->getStringArgument("OutputPath");
   FatalIf(baseDirectory.substr(0, 7) != "output/","OutputPath must begin with \"output\"\n");

   auto fileManager = std::make_shared<FileManager> (mpiBlock, baseDirectory);
   return fileManager;
}

std::shared_ptr<WeightData> gatherWeightsByBlock(
      std::shared_ptr<WeightData const> weightData,
      std::shared_ptr<MPIBlock const> mpiBlock) {
   FatalIf(
         weightData->getNumArbors() != 1,
         "gatherWeightsByBlock() called with multiple arbors.\n");

   std::shared_ptr<WeightData> result;
   if (mpiBlock->getRank() == 0) {
      result = std::make_shared<WeightData>(
            weightData->getNumArbors(),
            weightData->getPatchSizeX() * mpiBlock->getNumColumns(),
            weightData->getPatchSizeY() * mpiBlock->getNumRows(),
            weightData->getPatchSizeF(),
            weightData->getNumDataPatchesX(),
            weightData->getNumDataPatchesY(),
            weightData->getNumDataPatchesF());
   }

   for (int p = 0; p < weightData->getNumDataPatchesOverall(); ++p) {
      Buffer<float> buffer(
            weightData->getDataFromDataIndex(0 /*arbor*/, p),
            weightData->getPatchSizeX(),
            weightData->getPatchSizeY(),
            weightData->getPatchSizeF());
      auto gatheredBuffer =
            BufferUtils::gather(mpiBlock, buffer, buffer.getWidth(), buffer.getHeight(), 0, 0);
      if (mpiBlock->getRank() == 0) {
         float *destPtr = result->getDataFromDataIndex(0 /*arbor*/, p);
         float *srcPtr  = gatheredBuffer.asVector().data();
         std::memcpy(destPtr, srcPtr, gatheredBuffer.getTotalElements() * sizeof(float));
      }
   }
   return result;
}

std::shared_ptr<WeightData> makeWeightData(
      Dimensions const &dimensions,
      std::shared_ptr<MPIBlock const> mpiBlock,
      float start,
      float step) {
   auto weightData = std::make_shared<WeightData>(
         1 /*numArbors*/,
         dimensions.nxPost, dimensions.nyPost, dimensions.nfPost,
         1 /*nxPre*/, 1 /*nyPre*/, dimensions.nfPre);
   setWeightDataValues(weightData, mpiBlock, start, step);
   return weightData;
}

void setWeightDataValues(
      std::shared_ptr<WeightData> weightData,
      std::shared_ptr<MPIBlock const> mpiBlock,
      float start,
      float step) {
   float *weightPtr = weightData->getData(0 /*arbor*/);
   int numDataPatches = weightData->getNumDataPatchesOverall();
   int patchSize = weightData->getPatchSizeOverall();
   int patchSizeX = weightData->getPatchSizeX();
   int patchSizeY = weightData->getPatchSizeY();
   int patchSizeF = weightData->getPatchSizeF();
   int numProcsX = mpiBlock->getGlobalNumColumns();
   int numProcsY = mpiBlock->getGlobalNumRows();
   int patchSizeGlobal = patchSize * numProcsX * numProcsY;
   int procIndexX = mpiBlock->getStartColumn() + mpiBlock->getColumnIndex();
   int procIndexY = mpiBlock->getStartRow() + mpiBlock->getRowIndex();
   for (int p = 0; p < numDataPatches; ++p) {
      for (int k = 0; k < patchSize; ++k) {
         // need to convert k, which is the location within the MPI block, to the global location
         int kx = kxPos(k, patchSizeX, patchSizeY, patchSizeF) + patchSizeX * procIndexX;
         int ky = kyPos(k, patchSizeX, patchSizeY, patchSizeF) + patchSizeY * procIndexY;
         int kf = featureIndex(k, patchSizeX, patchSizeY, patchSizeF);
         int kGlobal =
               kIndex(kx, ky, kf, patchSizeX * numProcsX, patchSizeY * numProcsY, patchSizeF);
         int globalIndex = p * patchSizeGlobal + kGlobal;
         int localIndex = p * patchSize + k;
         weightPtr[localIndex] = start + static_cast<float>(globalIndex) * step;
      }
   }
}

int readUsingBroadcastPreWeightsFileAndCheck(
      BroadcastPreWeightsFile &file,
      std::shared_ptr<WeightData> weightsInFile,
      Dimensions const &dimensions,
      std::shared_ptr<MPIBlock const> mpiBlock,
      float correctStart,
      float correctStep,
      double correctTimestamp) {
   auto correctWeights = makeWeightData(dimensions, mpiBlock, correctStart, correctStep);
   double observedTimestamp;
   file.read(observedTimestamp);
   int status = compareWeightsAndTimestamps(
         correctTimestamp, observedTimestamp, correctWeights, weightsInFile,
         "readUsingBroadcastPreWeightsFileAndCheck()");
   return status;
}

int readUsingFileStreamAndCheck(
      std::shared_ptr<FileManager const> fileManager,
      std::string const &path,
      Dimensions const &dimensions,
      std::shared_ptr<MPIBlock const> mpiBlock,
      float correctStart,
      float correctStep,
      double correctTimestamp,
      int frameNumber) {
   auto correctWeights = makeWeightData(dimensions, mpiBlock, correctStart, correctStep);
   double observedTimestamp;
   std::shared_ptr<WeightData> weightsInFile = readUsingFileStreamPrimitives(
         fileManager, path, dimensions, frameNumber, observedTimestamp);
   int status = compareWeightsAndTimestamps(
         correctTimestamp, observedTimestamp, correctWeights, weightsInFile,
         "readUsingFileStreamAndCheck()");
   return status;
}

std::shared_ptr<WeightData> readUsingFileStreamPrimitives(
      std::shared_ptr<FileManager const> fileManager,
      std::string const &path,
      Dimensions const &dimensions,
      int frameNumber,
      double &timestamp) {
   auto weightData = std::make_shared<WeightData>(
      1 /*numArbors*/,
      dimensions.nxPost, dimensions.nyPost, dimensions.nfPost,
      1 /*numDataPatchesX*/, 1 /*numDataPatchesY*/, dimensions.nfPre);
   auto fileStream = FileStreamBuilder(
      fileManager,
      path,
      false /*isTextFlag*/,
      true /*readOnlyFlag*/,
      false /*clobberFlag*/,
      false /*verifyWritesFlag*/).get();
   auto mpiBlock = fileManager->getMPIBlock();
   std::shared_ptr<WeightData> gatheredWeightData = nullptr;
   if (fileStream) {
      int numPatches = weightData->getNumDataPatchesOverall();
      int nxpBlock = weightData->getPatchSizeX() * mpiBlock->getNumColumns();
      int nypBlock = weightData->getPatchSizeY() * mpiBlock->getNumRows();
      int nfp      = weightData->getPatchSizeF();
      gatheredWeightData = std::make_shared<WeightData>(
            1 /*numArbors*/, nxpBlock, nypBlock, nfp, 1, 1, numPatches);
      int frameSize = calcFrameSize(weightData, mpiBlock);
      long int filePosition = static_cast<long int>(frameSize) * static_cast<long int>(frameNumber);
      fileStream->setInPos(filePosition, std::ios_base::beg);
      fileStream->setOutPos(filePosition, std::ios_base::beg);
      BufferUtils::WeightHeader weightHeader;
      fileStream->read(&weightHeader, 104L);
      timestamp = weightHeader.baseHeader.timestamp;
      short int nx;
      short int ny;
      int offset;
      short int nxCorrect = static_cast<short int>(gatheredWeightData->getPatchSizeX());
      short int nyCorrect = static_cast<short int>(gatheredWeightData->getPatchSizeY());

      for (int p = 0; p < numPatches; ++p) {
         fileStream->read(&nx, 2);
         fileStream->read(&ny, 2);
         fileStream->read(&offset, 4);
         FatalIf(
               nx != nxCorrect or ny != nyCorrect or offset != 0,
               "Patch %d, patch header (nx=%hd,ny=%hd,offset=%d) "
               "does not match expected (%hd,%hd,%d)\n",
               p, nx, ny, offset, nxCorrect, nyCorrect, 0);
         fileStream->read(
               gatheredWeightData->getDataFromDataIndex(0 /*arbor*/, p),
               gatheredWeightData->getPatchSizeOverall() * sizeof(float));
      }
   }

   MPI_Bcast(&timestamp, 1, MPI_DOUBLE, 0, mpiBlock->getComm());
   scatterWeightsByBlock(weightData, gatheredWeightData, mpiBlock);
   return weightData;
}

void scatterWeightsByBlock(
      std::shared_ptr<WeightData> weightData,
      std::shared_ptr<WeightData const> gatheredWeightData,
      std::shared_ptr<MPIBlock const> mpiBlock) {
   FatalIf(
         weightData->getNumArbors() != 1,
         "scatterWeightsByBlock() called with multiple arbors\n");
   for (int p = 0; p < weightData->getNumDataPatchesOverall(); ++p) {
      Buffer<float> buffer;
      if (mpiBlock->getRank() == 0) {
         buffer.set(
               gatheredWeightData->getDataFromDataIndex(0 /*arbor*/, p), 
               gatheredWeightData->getPatchSizeX(),
               gatheredWeightData->getPatchSizeY(),
               gatheredWeightData->getPatchSizeF());
      }
      else {
         buffer.resize(
               weightData->getPatchSizeX(),
               weightData->getPatchSizeY(),
               weightData->getPatchSizeF());
      }
      int batchDimension = mpiBlock->getBatchDimension();
      for (int mpiBatchIndex = 0; mpiBatchIndex < batchDimension; ++mpiBatchIndex) {
         BufferUtils::scatter(
               mpiBlock,
               buffer,
               weightData->getPatchSizeX(),
               weightData->getPatchSizeY(),
               mpiBatchIndex,
               0 /*sourceProcess*/);
      }
      assert(buffer.getWidth() == weightData->getPatchSizeX());
      assert(buffer.getHeight() == weightData->getPatchSizeY());
      assert(buffer.getFeatures() == weightData->getPatchSizeF());
      std::memcpy(
            weightData->getDataFromDataIndex(0 /*arbor*/, p),
            buffer.asVector().data(),
            static_cast<std::size_t>(weightData->getPatchSizeOverall()) * sizeof(float));
   }
}

int verifyFrameIndex(BroadcastPreWeightsFile &file, int correctFrame, std::string filename) {
   int status = PV_SUCCESS;
   int currentFrame = file.getIndex();
   if (currentFrame != correctFrame) {
      ErrorLog().printf(
            "After writing to \"%s\", frame index was %d instead of expected %d\n",
            filename.c_str(), currentFrame, correctFrame);
      status = PV_FAILURE;
   }
   return status;
}

void writeUsingFileStreamPrimitives(
      std::shared_ptr<FileManager const> fileManager,
      std::string const &path,
      std::shared_ptr<WeightData const> weightData,
      double timestamp,
      int frameNumber) {
   FatalIf(
         weightData->getNumArbors() != 1,
         "writeUsingFileStreamPrimitives() called with multiple arbors.\n");

   auto fileStream = FileStreamBuilder(
      fileManager,
      path,
      false /*isTextFlag*/,
      false /*readOnlyFlag*/,
      false /*clobberFlag*/,
      false /*verifyWritesFlag*/).get();
   auto mpiBlock = fileManager->getMPIBlock();
   auto gatheredWeightData = gatherWeightsByBlock(weightData, mpiBlock);
   if (fileStream) {
      pvAssert(gatheredWeightData->getNumArbors() == weightData->getNumArbors());
      float minWeight = gatheredWeightData->getData(0)[0];
      float maxWeight = gatheredWeightData->getData(0)[0];
      long int numWeights = gatheredWeightData->getNumValuesPerArbor();
      for (int k = 0; k < numWeights; ++k) {
         float w = gatheredWeightData->getData(0 /*arbor*/)[k];
         minWeight = w < minWeight ? w : minWeight;
         maxWeight = w > maxWeight ? w : maxWeight;
      }
      int nxpBlock   = gatheredWeightData->getPatchSizeX();
      int nypBlock   = gatheredWeightData->getPatchSizeY();
      int nfp        = gatheredWeightData->getPatchSizeF();
      int numPatches = gatheredWeightData->getNumDataPatchesOverall();

      int frameSize = calcFrameSize(gatheredWeightData, nullptr /*mpiBlock*/);
      long int filePosition = static_cast<long int>(frameSize) * static_cast<long int>(frameNumber);
      fileStream->setInPos(filePosition, std::ios_base::beg);
      fileStream->setOutPos(filePosition, std::ios_base::beg);

      BufferUtils::WeightHeader weightHeader = BufferUtils::buildWeightHeader(
            false /*sharedFlag*/,
            gatheredWeightData->getNumDataPatchesX(),
            gatheredWeightData->getNumDataPatchesY(),
            gatheredWeightData->getNumDataPatchesF(),
            gatheredWeightData->getNumDataPatchesX(),
            gatheredWeightData->getNumDataPatchesY(),
            gatheredWeightData->getNumArbors(),
            timestamp,
            nxpBlock,
            nypBlock,
            nfp,
            false /*compressFlag*/,
            minWeight /*minVal*/,
            maxWeight /*maxVal*/);
      fileStream->write(&weightHeader, sizeof(weightHeader));
      char patchHeader[8];
      short int nx = static_cast<short int>(gatheredWeightData->getPatchSizeX());
      short int ny = static_cast<short int>(gatheredWeightData->getPatchSizeY());
      int offset = 0;
      std::memcpy(&patchHeader[0], &nx, 2);
      std::memcpy(&patchHeader[2], &ny, 2);
      std::memcpy(&patchHeader[4], &offset, 4);
      
      for (int p = 0; p < gatheredWeightData->getNumDataPatchesOverall(); ++p) {
         fileStream->write(patchHeader, 8);
         fileStream->write(
               gatheredWeightData->getDataFromDataIndex(0 /*arbor*/, p),
               gatheredWeightData->getPatchSizeOverall() * sizeof(float));
      }
      fileStream->setInPos(filePosition + frameSize, std::ios_base::beg);
      fileStream->setOutPos(filePosition + frameSize, std::ios_base::beg);
   }
}
