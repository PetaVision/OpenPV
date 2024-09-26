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
   auto mpiBlock = communicator->getIOMPIBlock();
   int blockSize = mpiBlock->getSize();
   int globalSize = communicator->globalCommSize();
   FatalIf(
         blockSize != globalSize,
         "%s must be run with only one MPIBlock\n"
         "    (CheckpointCellNumRows == NumRows, CheckpointCellNumColumns == NumColumns, "
         "    CheckpointCellBatchDimension == BatchWidth)\n",
         programName.c_str());
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

void checkFrame(std::shared_ptr<FileStream> fileStream) {
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
      fileStream->read(readValues[i].data(), numBytes);
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
            fileStream.get());
   }
}
