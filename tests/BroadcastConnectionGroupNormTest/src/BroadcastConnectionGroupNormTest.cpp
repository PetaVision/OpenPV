/*
 * BroadcastConnectionGroupNormTest.cpp
 */

#include <columns/buildandrun.hpp>
#include <io/FileStreamBuilder.hpp>
#include <structures/MPIBlock.hpp>
#include <utils/BufferUtilsPvp.hpp>
#include <utils/PathComponents.hpp> // baseName
#include <utils/PVAssert.hpp>
#include <utils/PVLog.hpp>
#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <unistd.h>

using namespace PV;

int checkOutput(HyPerCol *hc, int argc, char *argv[]);
std::vector<float> checkFrame(std::shared_ptr<FileStream> fileStream);
void checkValues(std::vector<std::shared_ptr<FileStream>> fileStreams, bool isRoot);

int main(int argc, char *argv[]) {
   FatalIf(sizeof(float) != 4UL, "float has size %zu, instead of required %lu\n", sizeof(int), 4UL);
   FatalIf(sizeof(int) != 4UL, "int has size %zu, instead of required %lu\n", sizeof(int), 4UL);
   FatalIf(
         sizeof(short int) != 2UL,
         "short int has size %zu, instead of required %lu\n",
         sizeof(int), 2UL);
   int status = buildandrun(argc, argv, nullptr /*custominit*/, checkOutput);
   if (status == PV_SUCCESS) {
      InfoLog() << "Test passed.\n";
      return EXIT_SUCCESS;
   }
   else {
      InfoLog() << "Test failed.\n";
      return EXIT_FAILURE;
   }
}

int checkOutput(HyPerCol *hc, int argc, char *argv[]) {
   auto *communicator = hc->getCommunicator();
   std::string programName = baseName(argv[0]);
   int globalSize = communicator->globalCommSize();
   int numRows = communicator->numCommRows();
   int numCols = communicator->numCommColumns();
   int batchWidth = communicator->numCommBatches();
   FatalIf(
         globalSize != 4 or numRows != 2 or numCols != 2 or batchWidth != 1,
         "%s must be run with 2 rows, 2 columns, and 1 batch element (given %d-by-%d-by-%d)\n",
         programName.c_str(), numRows, numCols, batchWidth);
   auto mpiBlock = communicator->getIOMPIBlock();
   std::string outputDir = hc->getOutputPath();
   auto outputFileManager = std::make_shared<FileManager>(mpiBlock, outputDir);
   std::vector<std::shared_ptr<FileStream>> connectionFileStreams(3);
   for (int i = 0; i < 3; ++i) {
      std::string filename("InputToOutput");
      filename.append(std::to_string(i)).append(".pvp");
      connectionFileStreams[i] = FileStreamBuilder(
            outputFileManager,
            filename,
            false /* isTextFlag */,
            true /* readOnlyFlag */,
            false /* clobberFlag */,
            false /* verifyWritesFlag */).get();
   }
   checkValues(connectionFileStreams, outputFileManager->isRoot());

   std::string lastCheckpointDir = hc->getLastCheckpointDir();
   auto checkpointFileManager = std::make_shared<FileManager>(mpiBlock, lastCheckpointDir);
   for (int i = 0; i < 3; ++i) {
      if (connectionFileStreams[i]) { connectionFileStreams[i]->close(); }
      // Is closing necessary? Reassigning the shared pointer should automatically deallocate the old value

      std::string filename("InputToOutput");
      filename.append(std::to_string(i)).append("_W.pvp");
      connectionFileStreams[i] = FileStreamBuilder(
            checkpointFileManager,
            filename,
            false /* isTextFlag */,
            true /* readOnlyFlag */,
            false /* clobberFlag */,
            false /* verifyWritesFlag */).get();
   }
   checkValues(connectionFileStreams, outputFileManager->isRoot());
   return EXIT_SUCCESS;
}

std::vector<float> checkFrame(std::shared_ptr<FileStream> fileStream) {
   int const nxpCorrect = 4;
   int const nypCorrect = 4;
   int const nfpCorrect = 1;
   int const numPatchesCorrect = 4;
   FatalIf(!fileStream, "checkFrame called with null fileStream");
   BufferUtils::WeightHeader header;
   FatalIf(
         sizeof(header) != 104UL,
         "WeightHeader size is %zu bytes, instead of expected %lu\n",
         sizeof(header), 104UL);
   fileStream->read(&header, 8L);
   FatalIf(
         header.baseHeader.headerSize != static_cast<int>(sizeof(header)),
         "checkValues(): headerSize is %d instead of the expected value %d\n",
         header.baseHeader.headerSize, static_cast<int>(sizeof(header)));
   FatalIf(header.baseHeader.numParams * 4 != header.baseHeader.headerSize,
         "checkValues(): numParams is %d instead of the expected value %d\n",
         header.baseHeader.numParams, static_cast<int>(sizeof(header)/4UL));
   fileStream->setInPos(-8L, std::ios_base::cur);
   fileStream->read(&header, static_cast<long>(sizeof(header)));
   FatalIf(
         header.baseHeader.numRecords != 1,
         "numRecords (number of arbors) is %d instead of the expected value %d\n",
         header.numPatches, 1);
   FatalIf(
         header.nxp != nxpCorrect,
         "nxp is %d instead of the expected value %d\n",
         header.nxp, nxpCorrect);
   FatalIf(
         header.nyp != nypCorrect,
         "nyp is %d instead of the expected value %d\n",
         header.nyp, nypCorrect);
   FatalIf(
         header.nfp != nfpCorrect,
         "nfp is %d instead of the expected value %d\n",
         header.nfp, nfpCorrect);
   FatalIf(
         header.numPatches != numPatchesCorrect,
         "numPatches is %d instead of the expected value %d\n",
         header.numPatches, 16);
   std::vector<std::vector<float>> readValues(header.numPatches);
   for (int i = 0; i < header.numPatches; ++i) {
      short int sPatchSize[2];
      fileStream->read(&sPatchSize, 4L);
      int nx = static_cast<int>(sPatchSize[0]);
      int ny = static_cast<int>(sPatchSize[1]);
      FatalIf(
            nx != nxpCorrect or ny != nypCorrect,
            "Patch size data for patch %d is %u-by-%u instead of the expected %d-by-%d\n",
            i, (int)sPatchSize[0], (int)sPatchSize[1], nxpCorrect, nypCorrect);
      int offset;
      fileStream->read(&offset, 4L);
      FatalIf(offset != 0, "Offset for patch %d is %d instead of the expected zero.\n", offset);
      readValues[i].resize(nx * ny);
      long numBytes = nx * ny * 4L; // size of float is 4.
      fileStream->read(readValues[i].data(), 64);
   }

   std::vector<float> sumOfSquares(4);
   for (int i = 0 ; i < header.numPatches; ++i) {
      float s = 0.0f;
      for (auto const &x : readValues[i]) { s += x*x; }
      sumOfSquares[i % 4] += s;
   }
   return sumOfSquares;
}

void checkValues(std::vector<std::shared_ptr<FileStream>> fileStreams, bool isRoot) {
   FatalIf(
         static_cast<unsigned int>(fileStreams.size()) != 3U,
         "checkValues(): fileStreams has size %u instead of the expected %u\n",
         static_cast<unsigned int>(fileStreams.size()), 3U);
   if (!isRoot) {
      for (int connectionIndex = 0; connectionIndex < 3; ++connectionIndex) {
         auto fileStream = fileStreams[connectionIndex];
         FatalIf(
               fileStream != nullptr,
               "Non-root ranks should have a null FileStream pointer (value at index %d is %p)\n",
               connectionIndex, fileStream.get());
      }
      return;
   }

   for (int connectionIndex = 0; connectionIndex < 3; ++connectionIndex) {
      auto fileStream = fileStreams[connectionIndex];
      FatalIf(
            fileStream == nullptr,
            "Root rank should have non-null FileStream pointers, but index %d is null.\n",
            connectionIndex);
   }
   std::vector<long int> filesizes(3);
   for (int connectionIndex = 0; connectionIndex < 3; ++connectionIndex) {
      auto fileStream = fileStreams[connectionIndex];
      fileStream->setInPos(0L, std::ios_base::end);
      filesizes[connectionIndex] = fileStream->getInPos();
      fileStream->setInPos(0L, std::ios_base::beg);
   }
   FatalIf(
         filesizes[1] != filesizes[0] or filesizes[2] != filesizes[0],
         "File sizes are not all the same:\n  %s -> %lu\n  %s -> %lu\n  %s -> %lu\n",
         fileStreams[0]->getFileName().c_str(), filesizes[0],
         fileStreams[1]->getFileName().c_str(), filesizes[1],
         fileStreams[2]->getFileName().c_str(), filesizes[2]);

   while (fileStreams[0]->getInPos() < filesizes[0]) {
      std::vector<float> sumOfSquares(4);

      for (int connectionIndex = 0; connectionIndex < 3; ++connectionIndex) {
         auto fileStream = fileStreams[connectionIndex];
         auto sumOfSquaresOneConn = checkFrame(fileStream);
         unsigned int sz = static_cast<unsigned int>(sumOfSquaresOneConn.size());
         FatalIf(
               sz != 4U,
               "checkFrame() returned a vector of size %u, instead of the expected 4\n",
               sz);
         for (int index = 0; index < 4; ++index) {
            sumOfSquares[index] += sumOfSquaresOneConn[index];
         }
      }

      int status = PV_SUCCESS;
      for (int i = 0; i < 4; ++i) {
         float s = sumOfSquares[i];
         if (std::fabs(s - 16.0f) > 4.0e-6f) {
            ErrorLog().printf(
                  "sum of squares for feature %d is %f instead of correct value 16 (discrepancy %g)\n",
                  i, static_cast<double>(s), static_cast<double>(s - 16.0f));
            status = PV_FAILURE;
         }
      }
      FatalIf(status != PV_SUCCESS, "Test failed.\n");
   }
}
