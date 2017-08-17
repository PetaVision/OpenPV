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
      std::string const &checkpointDirectory,
      double simTime,
      bool verifyWritesFlag) const {
   std::string path(checkpointDirectory);
   path.append("/").append(getName()).append(".pvp");
   FileStream *fileStream = nullptr;
   if (getMPIBlock()->getRank() == 0) {
      fileStream = new FileStream(path.c_str(), std::ios_base::out, verifyWritesFlag);
   }

   WeightsFileIO weightFileIO(fileStream, getMPIBlock(), mWeights);
   weightFileIO.writeWeights(simTime, mCompressFlag);
   delete fileStream;
}

void CheckpointEntryWeightPvp::read(std::string const &checkpointDirectory, double *simTimePtr)
      const {
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

   std::string path(checkpointDirectory);
   path.append("/").append(getName()).append(".pvp");
   FileStream *fileStream = nullptr;
   if (getMPIBlock()->getRank() == 0) {
      fileStream = new FileStream(path.c_str(), std::ios_base::in, false);
   }

   WeightsFileIO weightFileIO(fileStream, getMPIBlock(), mWeights);
   double simTime = weightFileIO.readWeights(0 /*frameNumber*/);
   if (simTimePtr) {
      *simTimePtr = simTime;
   }
   delete fileStream;
}

void CheckpointEntryWeightPvp::remove(std::string const &checkpointDirectory) const {
   deleteFile(checkpointDirectory, "pvp");
}

} // end namespace PV
