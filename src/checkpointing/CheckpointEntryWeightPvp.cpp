/*
 * CheckpointEntry.cpp
 *
 *  Created on Sep 27, 2016
 *      Author: Pete Schultz
 */

#include "CheckpointEntryWeightPvp.hpp"
#include "io/fileio.hpp"
#include "io/BroadcastPreWeightsFile.hpp"
#include "io/LocalPatchWeightsFile.hpp"
#include "io/SharedWeightsFile.hpp"
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
   std::shared_ptr<WeightsFile> weightsFile;

   switch(mWeights->getWeightsType()) {
      case Weights::WeightsType::SHARED:
         weightsFile = std::make_shared<SharedWeightsFile>(
            fileManager,
            filename,
            mWeights->getData(),
            mCompressFlag,
            false /*readOnlyFlag*/,
            true /*clobberFlag*/,
            verifyWritesFlag);
         break;
      case Weights::WeightsType::LOCALPATCH:
         weightsFile = std::make_shared<LocalPatchWeightsFile>(
               fileManager,
               filename,
               mWeights->getData(),
               &mWeights->getGeometry()->getPreLoc(),
               &mWeights->getGeometry()->getPostLoc(),
               true /*fileExtendedFlag*/,
               mCompressFlag,
               false /*readOnlyFlag*/,
               true /*clobberFlag*/,
               verifyWritesFlag);
         break;
      case Weights::WeightsType::BROADCASTPRE:
         weightsFile = std::make_shared<BroadcastPreWeightsFile>(
               fileManager,
               filename,
               mWeights->getData(),
               mWeights->getGeometry()->getPreLoc().nf,
               mCompressFlag,
               false /*readOnlyFlag*/,
               true /*clobberFlag*/,
               verifyWritesFlag);
         break;
      default:
         Fatal().printf("Unrecognized WeightsType %d\n", mWeights->getWeightsType());
         break;
   }
   weightsFile->write(simTime);
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
   std::shared_ptr<WeightsFile> weightsFile;

   switch(mWeights->getWeightsType()) {
      case Weights::WeightsType::SHARED:
         weightsFile = std::make_shared<SharedWeightsFile>(
            fileManager,
            filename,
            mWeights->getData(),
            mCompressFlag,
            true /*readOnlyFlag*/,
            false /*clobberFlag*/,
            false /*verifyWritesFlag*/);
         break;
      case Weights::WeightsType::LOCALPATCH:
         weightsFile = std::make_shared<LocalPatchWeightsFile>(
               fileManager,
               filename,
               mWeights->getData(),
               &mWeights->getGeometry()->getPreLoc(),
               &mWeights->getGeometry()->getPostLoc(),
               true /*fileExtendedFlag*/,
               mCompressFlag,
               true /*readOnlyFlag*/,
               false /*clobberFlag*/,
               false /*verifyWritesFlag*/);
         break;
      case Weights::WeightsType::BROADCASTPRE:
         weightsFile = std::make_shared<BroadcastPreWeightsFile>(
               fileManager,
               filename,
               mWeights->getData(),
               mWeights->getGeometry()->getPreLoc().nf,
               mCompressFlag,
               true /*readOnlyFlag*/,
               false /*clobberFlag*/,
               false /*verifyWritesFlag*/);
         break;
      default:
         Fatal().printf("Unrecognized WeightsType %d\n", mWeights->getWeightsType());
         break;
   }
   double simTime;
   weightsFile->read(simTime);
   if (simTimePtr) {
      *simTimePtr = simTime;
   }
}

void CheckpointEntryWeightPvp::remove(std::shared_ptr<FileManager const> fileManager) const {
   deleteFile(fileManager, std::string("pvp"));
}

} // end namespace PV
