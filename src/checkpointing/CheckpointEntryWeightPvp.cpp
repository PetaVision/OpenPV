/*
 * CheckpointEntry.cpp
 *
 *  Created on Sep 27, 2016
 *      Author: Pete Schultz
 */

#include "CheckpointEntryWeightPvp.hpp"
#include "io/WeightsFileIO.hpp"
#include "io/fileio.hpp"
#include "structures/Buffer.hpp"
#include "utils/BufferUtilsMPI.hpp"
#include "utils/BufferUtilsPvp.hpp"
#include <limits>

namespace PV {

void CheckpointEntryWeightPvp::initialize(Weights *weights, bool compressFlag) {
   mWeights      = weights;
   mCompressFlag = compressFlag;
}

void CheckpointEntryWeightPvp::write(
      std::shared_ptr<FileManager const> fileManager,
      double simTime,
      bool verifyWritesFlag) const {
   std::string filename = generateFilename(std::string("pvp"));
   
   auto fileStream = fileManager->open(filename, std::ios_base::out, verifyWritesFlag);
   WeightsFileIO weightFileIO(fileStream.get(), fileManager->getMPIBlock(), mWeights);
   weightFileIO.writeWeights(simTime, mCompressFlag);
}

void CheckpointEntryWeightPvp::read(
      std::shared_ptr<FileManager const> fileManager, double *simTimePtr) const {
   // Need to clear weights before reading because reading weights is increment-add, not assignment.
   int const numArbors = mWeights->getNumArbors();
   for (int arbor = 0; arbor < numArbors; arbor++) {
      int const nxp        = mWeights->getPatchSizeX();
      int const nyp        = mWeights->getPatchSizeY();
      int const nfp        = mWeights->getPatchSizeF();
      int const numPatches = mWeights->getNumDataPatches();

      std::size_t const numWeightsInArbor = (std::size_t)(numPatches * nxp * nyp * nfp);
      float *weightData                   = mWeights->getData(arbor);

      memset(weightData, 0, numWeightsInArbor * sizeof(*weightData));
   }

   std::string filename = generateFilename(std::string("pvp"));
   auto fileStream = fileManager->open(filename, std::ios_base::in, false);

   WeightsFileIO weightFileIO(fileStream.get(), fileManager->getMPIBlock(), mWeights);
   double simTime = weightFileIO.readWeights(0 /*frameNumber*/);
   if (simTimePtr) {
      *simTimePtr = simTime;
   }
}

void CheckpointEntryWeightPvp::remove(std::shared_ptr<FileManager const> fileManager) const {
   deleteFile(fileManager, std::string("pvp"));
}

} // end namespace PV
