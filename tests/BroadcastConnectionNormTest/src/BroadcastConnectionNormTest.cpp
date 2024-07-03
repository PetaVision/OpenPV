/*
 * BroadcastConnectionNormTest.cpp
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

using namespace PV;

int checkOutput(HyPerCol *hc, int argc, char *argv[]);
void checkFrame(std::shared_ptr<FileStream> fileStream);
void checkValues(std::shared_ptr<FileStream> fileStream, bool isRoot);

int main(int argc, char *argv[]) {
   FatalIf(sizeof(float) != 4UL, "float has size %zu, instead of required %lu\n", sizeof(int), 4UL);
   FatalIf(sizeof(int) != 4UL, "int has size %zu, instead of required %lu\n", sizeof(int), 4UL);
   FatalIf(
         sizeof(short int) != 2UL,
         "short int has size %zu, instead of required %lu\n",
         sizeof(int), 2UL);
   int status = buildandrun(argc, argv, nullptr /*custominit*/, checkOutput);
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
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
         programName, numRows, numCols, batchWidth);
   auto mpiBlock = communicator->getIOMPIBlock();
   std::string outputDir = hc->getOutputPath();
   auto outputFileManager = std::make_shared<FileManager>(mpiBlock, outputDir);
   auto outputFileStream = FileStreamBuilder(
         outputFileManager,
         "InputToOutput.pvp",
         false /* isTextFlag */,
         true /* readOnlyFlag */,
         false /* clobberFlag */,
         false /* verifyWritesFlag */).get();
   checkValues(outputFileStream, outputFileManager->isRoot());
   outputFileStream = nullptr;

   std::string lastCheckpointDir = hc->getLastCheckpointDir();
   auto checkpointFileManager = std::make_shared<FileManager>(mpiBlock, lastCheckpointDir);
   auto checkpointFileStream = FileStreamBuilder(
         checkpointFileManager,
         "InputToOutput_W.pvp",
         false /* isTextFlag */,
         true /* readOnlyFlag */,
         false /* clobberFlag */,
         false /* verifyWritesFlag */).get();
   checkValues(checkpointFileStream, outputFileManager->isRoot());
   return EXIT_SUCCESS;
}

// Note: because of a (temporary) hack to implement broadcast layers, the weight files currently
// have dimensions nxp=2, nyp=2, nfp=1, numPatches = 16. Ideally, they would have dimensions
// nxp=4, nyp=4, nfp=1, numPatches=4. When this change is implemented, the code below will need
// to be adjusted. --Pete Schultz, Jul 2, 2024.
void checkFrame(std::shared_ptr<FileStream> fileStream) {
   int const nxpCorrect = 2;
   int const nypCorrect = 2;
   int const nfpCorrect = 1;
   int const numPatchesCorrect = 16;
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
         header.numPatches != 16,
         "numPatches is %d instead of the expected value %d\n",
         header.numPatches, 16);
   int frameSize = (header.nxp * header.nyp * header.nyp + 8) * header.numPatches;
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
      fileStream->read(readValues[i].data(), 16L); // (nx * ny values, each of size sizeof(float))
   }

   std::vector<float> sumOfSquares(4);
   for (int i = 0 ; i < header.numPatches; ++i) {
      float s = 0.0f;
      for (auto const &x : readValues[i]) { s += x*x; }
      sumOfSquares[i % 4] += s;
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

void checkValues(std::shared_ptr<FileStream> fileStream, bool isRoot) {
   if (isRoot) {
      FatalIf(
            fileStream == nullptr,
            "Root rank should have a non-null FileStream pointer, but it is null.\n");
      fileStream->setInPos(0L, std::ios_base::end);
      long int eof = fileStream->getInPos();
      fileStream->setInPos(0L, std::ios_base::beg);
      while (fileStream->getInPos() < eof) {
         checkFrame(fileStream);
      }
   }
   else {
      FatalIf(
            fileStream != nullptr,
            "Non-root ranks should have a null FileStream pointer (value is %p)\n",
            fileStream);
   }
}
