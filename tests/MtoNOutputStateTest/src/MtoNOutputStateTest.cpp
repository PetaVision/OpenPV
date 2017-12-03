/*
 * MtoNOutputStateTest.cpp
 *
 */

#include "IndexLayer.hpp"
#include "IndexWeightConn.hpp"
#include <columns/PV_Init.hpp>
#include <columns/buildandrun.hpp>
#include <cstdlib>
#include <utils/BufferUtilsPvp.hpp>

void verifyOutputStateFiles(
      std::string const &blockDirectory,
      MPIBlock const &mpiBlock,
      int batchElementsPerProcess);
void verifyOutputStatePre(
      std::string const &path,
      MPIBlock const &mpiBlock,
      int batchElementsPerProcess);
void verifyOutputStatePost(
      std::string const &path,
      MPIBlock const &mpiBlock,
      int batchElementsPerProcess);
void verifyOutputStateSharedWeights(std::string const &path, MPIBlock const &mpiBlock);

int main(int argc, char *argv[]) {
   // Initialize
   PV_Init pv_init(&argc, &argv, false /*no unrecognized arguments*/);
   pv_init.registerKeyword("IndexLayer", Factory::create<IndexLayer>);
   pv_init.registerKeyword("IndexWeightConn", Factory::create<IndexWeightConn>);

   InfoLog() << "PID " << getpid() << ", global rank " << pv_init.getWorldRank() << ".\n";

   // Creating a checkpointer with the same constructor arguments as the
   // HyPerCol's checkpointer will have is the easiest way to get the
   // directory that outputState will write to, and the rank within the
   // MPIBlock.
   Checkpointer *tempCheckpoint = new Checkpointer(
         "column", pv_init.getCommunicator()->getGlobalMPIBlock(), pv_init.getArguments());
   tempCheckpoint->ioParams(PARAMS_IO_READ, pv_init.getParams());
   std::string outputDirectory(tempCheckpoint->getOutputPath());
   std::string const &blockDirectory = tempCheckpoint->getBlockDirectoryName();
   if (!blockDirectory.empty()) {
      outputDirectory.append("/").append(blockDirectory);
   }
   MPIBlock mpiBlock = *tempCheckpoint->getMPIBlock();

   int const rankInBlock = mpiBlock.getRank();
   delete tempCheckpoint;

   // If the directory already exists, delete it so that results aren't skewed
   // by results from a previous run.
   if (rankInBlock == 0) {
      std::string rmrfcommand = std::string("rm -rf \"") + outputDirectory + "\"";
      int rmrfstatus          = std::system(rmrfcommand.c_str());
      if (rmrfstatus > 0) {
         rmrfstatus = WEXITSTATUS(rmrfstatus);
      }
      FatalIf(
            rmrfstatus != 0,
            "Global rank %d failed running the command '%s' with exit value %d\n",
            pv_init.getWorldRank(),
            rmrfstatus);
   }

   HyPerCol *hc          = new HyPerCol(&pv_init);
   int processBatchWidth = hc->getNBatchGlobal() / mpiBlock.getGlobalBatchDimension();
   int status            = hc->run();

   if (rankInBlock == 0) {
      verifyOutputStateFiles(outputDirectory, mpiBlock, processBatchWidth);
   }

   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

void verifyOutputStateFiles(
      std::string const &blockDirectory,
      MPIBlock const &mpiBlock,
      int batchElementsPerProcess) {
   std::string outputStatePre = blockDirectory;
   if (!blockDirectory.empty()) {
      outputStatePre.append("/");
   }
   outputStatePre.append("Pre.pvp");
   verifyOutputStatePre(outputStatePre, mpiBlock, batchElementsPerProcess);

   std::string outputStatePost = blockDirectory;
   if (!blockDirectory.empty()) {
      outputStatePost.append("/");
   }
   outputStatePost.append("Post.pvp");
   verifyOutputStatePost(outputStatePost, mpiBlock, batchElementsPerProcess);

   std::string outputStateSharedWeights = blockDirectory;
   if (!blockDirectory.empty()) {
      outputStateSharedWeights.append("/");
   }
   outputStateSharedWeights.append("SharedWeights.pvp");
   verifyOutputStateSharedWeights(outputStateSharedWeights, mpiBlock);
}

void verifyOutputStatePre(
      std::string const &path,
      MPIBlock const &mpiBlock,
      int batchElementsPerProcess) {
   FileStream outputStatePreStream(path.c_str(), std::ios_base::in);
   auto header = BufferUtils::readActivityHeader(outputStatePreStream);
   FatalIf(header.fileType != PVP_NONSPIKING_ACT_FILE_TYPE, "%s is not a dense activity file.\n");
   FatalIf(header.dataType != BufferUtils::FLOAT, "%s does not have dataType=float.\n");

   int batchElementsPerBlock = batchElementsPerProcess * mpiBlock.getBatchDimension();

   int const numFrames  = header.nBands;
   int const numNeurons = header.nx * header.ny * header.nf;

   double frameTime    = std::strtod("NAN", nullptr);
   int localBatchIndex = 0;
   Buffer<float> buffer(header.nx, header.ny, header.nf);

   bool errorFound = false;
   for (int frame = 0; frame < numFrames; frame++) {
      double newTime  = BufferUtils::readFrame<float>(outputStatePreStream, &buffer);
      localBatchIndex = (newTime == frameTime) ? localBatchIndex + 1 : 0;
      if (localBatchIndex != frame % batchElementsPerBlock) {
         ErrorLog().printf(
               "%s, frame %d appears to be local batch index %d of time %f; expected local batch "
               "index %d of time %f.\n",
               path.c_str(),
               frame,
               localBatchIndex,
               newTime,
               frame % batchElementsPerBlock,
               (double)(frame / batchElementsPerBlock)); // Integer division
      }
      frameTime = newTime;
      for (int k = 0; k < numNeurons; k++) {
         int x          = kxPos(k, header.nx, header.ny, header.nf);
         int y          = kyPos(k, header.nx, header.ny, header.nf);
         int f          = featureIndex(k, header.nx, header.ny, header.nf);
         float observed = buffer.at(x, y, f);
         float expected = 1.0f;
         if (observed != expected) {
            ErrorLog().printf(
                  "%s, frame %d, t = %f, local batch index = %d, global batch index %d, x = %d, y "
                  "= %d, f = %d: expected %f, observed %f.\n",
                  path.c_str(),
                  frame,
                  frameTime,
                  localBatchIndex,
                  localBatchIndex + mpiBlock.getStartBatch() * batchElementsPerProcess,
                  x,
                  y,
                  f,
                  (double)expected,
                  (double)observed);
            errorFound = true;
         }
      }
   }
   FatalIf(errorFound, "Error in Pre.pvp\n");
}

void verifyOutputStatePost(
      std::string const &path,
      MPIBlock const &mpiBlock,
      int batchElementsPerProcess) {
   FileStream outputStatePostStream(path.c_str(), std::ios_base::in);
   auto header = BufferUtils::readActivityHeader(outputStatePostStream);
   FatalIf(header.fileType != PVP_NONSPIKING_ACT_FILE_TYPE, "%s is not a dense activity file.\n");
   FatalIf(header.dataType != BufferUtils::FLOAT, "%s does not have dataType=float.\n");

   int batchElementsPerBlock = batchElementsPerProcess * mpiBlock.getBatchDimension();

   int const numFrames = header.nBands;

   int numNeurons       = header.nx * header.ny * header.nf;
   int nxGlobal         = header.nx * (mpiBlock.getGlobalNumColumns() / mpiBlock.getNumColumns());
   int nyGlobal         = header.ny * (mpiBlock.getGlobalNumRows() / mpiBlock.getNumRows());
   int numNeuronsGlobal = nxGlobal * nyGlobal * header.nf;
   int kx0              = header.nx * (mpiBlock.getStartColumn() / mpiBlock.getNumColumns());
   int ky0              = header.ny * (mpiBlock.getStartRow() / mpiBlock.getNumRows());

   double frameTime    = std::strtod("NAN", nullptr);
   int localBatchIndex = 0;
   Buffer<float> buffer(header.nx, header.ny, header.nf);

   bool errorFound = false;
   for (int frame = 0; frame < numFrames; frame++) {
      double newTime  = BufferUtils::readFrame<float>(outputStatePostStream, &buffer);
      localBatchIndex = (newTime == frameTime) ? localBatchIndex + 1 : 0;
      if (localBatchIndex != frame % batchElementsPerBlock) {
         ErrorLog().printf(
               "%s, frame %d appears to be local batch index %d of time %f; expected local batch "
               "index %d of time %f.\n",
               path.c_str(),
               frame,
               localBatchIndex,
               newTime,
               frame % batchElementsPerBlock,
               (double)(frame / batchElementsPerBlock)); // Integer division
      }
      frameTime = newTime;
      for (int k = 0; k < numNeurons; k++) {
         int x                = kxPos(k, header.nx, header.ny, header.nf);
         int y                = kyPos(k, header.nx, header.ny, header.nf);
         int f                = featureIndex(k, header.nx, header.ny, header.nf);
         float observed       = buffer.at(x, y, f);
         int batchIndexOffset = mpiBlock.getStartBatch() * batchElementsPerProcess;
         int globalBatchIndex = localBatchIndex + batchIndexOffset;
         int xGlobal          = x + kx0;
         int yGlobal          = y + ky0;
         int kGlobal          = kIndex(x + kx0, y + ky0, f, nxGlobal, nyGlobal, header.nf);
         float expected = (float)frameTime * (float)(globalBatchIndex * numNeuronsGlobal + kGlobal);
         if (observed != expected) {
            ErrorLog().printf(
                  "%s, frame %d, t = %f, local batch index = %d, global batch index %d, local "
                  "coordinates (x,y,f) = (%d,%d,%d), global coordinates (x,y,f) = (%d,%d,%d), "
                  "local index %d, global index %d: expected %f, observed %f.\n",
                  path.c_str(),
                  frame,
                  frameTime,
                  localBatchIndex,
                  globalBatchIndex,
                  x,
                  y,
                  f,
                  xGlobal,
                  yGlobal,
                  f,
                  k,
                  kGlobal,
                  (double)expected,
                  (double)observed);
            errorFound = true;
         }
      }
   }
   FatalIf(errorFound, "Error in Post.pvp\n");
}

void verifyOutputStateSharedWeights(std::string const &path, MPIBlock const &mpiBlock) {

   // FileStream does not make the file size available, but calls Fatal() if you read past the end,
   // so we need to get the file size.
   struct stat fileStat;
   int statstatus = stat(path.c_str(), &fileStat);
   FatalIf(statstatus != 0, "stat \"%s\" failed: %s\n", path.c_str(), strerror(errno));
   FatalIf(!S_ISREG(fileStat.st_mode), "\"%s\" is not a regular file.\n", path.c_str());
   long const fileSize = (long)fileStat.st_size;

   FileStream fileStream(path.c_str(), std::ios_base::in);

   bool errorFound = false;
   while (fileStream.getInPos() < fileSize) {
      long frameStart = fileStream.getInPos();
      BufferUtils::WeightHeader weightHeader;
      fileStream.read(&weightHeader, sizeof(weightHeader));
      FatalIf(
            weightHeader.baseHeader.fileType != PVP_KERNEL_FILE_TYPE,
            "%s is not a shared-weights file.\n");
      FatalIf(
            weightHeader.baseHeader.dataType != BufferUtils::FLOAT,
            "%s does not have dataType=float.\n");

      int const nxp = weightHeader.nxp;
      int const nyp = weightHeader.nyp;
      int const nfp = weightHeader.nfp;
      int patchSize = nxp * nyp * nfp;
      std::vector<float> weightBuffer(patchSize);
      int const numPatches = weightHeader.numPatches;
      int const numArbors  = weightHeader.baseHeader.nBands;

      for (int a = 0; a < numArbors; a++) {
         for (int p = 0; p < numPatches; p++) {
            long patchStart = fileStream.getInPos();
            short patchDims[2];
            fileStream.read(patchDims, (size_t)2 * sizeof(*patchDims));
            int offset;
            fileStream.read(&offset, sizeof(offset));
            FatalIf(
                  (int)patchDims[0] != nxp,
                  "Shared-weights file %s: frame starting at %ld, "
                  "arbor %d, patch %d: nx is not header's nxp. (%d vs %d)\n",
                  path.c_str(),
                  frameStart,
                  a,
                  p,
                  (int)patchDims[0],
                  nxp);
            FatalIf(
                  (int)patchDims[1] != nyp,
                  "Shared-weights file %s: frame starting at %ld, "
                  "arbor %d, patch %d: ny is not header's nyp. (%d vs %d)\n",
                  path.c_str(),
                  frameStart,
                  a,
                  p,
                  (int)patchDims[1],
                  nyp);
            FatalIf(
                  offset != 0,
                  "Shared-weights file %s: frame starting at %ld, arbor %d, patch "
                  "%d: offset is not 0.\n");

            fileStream.read(weightBuffer.data(), weightBuffer.size() * sizeof(float));
            for (int index = 0; index < patchSize; index++) {
               float observed = weightBuffer[index];
               float expected = (float)index + (float)weightHeader.baseHeader.timestamp;
               if (observed != expected) {
                  ErrorLog().printf(
                        "%s, frame starting at %ld, patch starting at %ld, index %d: expected %f, "
                        "observed %f.\n",
                        path.c_str(),
                        frameStart,
                        patchStart,
                        index,
                        (double)expected,
                        (double)observed);
               }
            }
         }
      }
      Buffer<float> buffer(weightHeader.nxp, weightHeader.nyp, weightHeader.nfp);
   }
   FatalIf(errorFound, "Error in SharedWeights.pvp\n");
}
