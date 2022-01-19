/*
 * CheckpointEntryRandState.cpp
 *
 *  Created on Oct 6, 2016
 *      Author: Pete Schultz
 */

#include "CheckpointEntryRandState.hpp"
#include "io/randomstateio.hpp"

namespace PV {

void CheckpointEntryRandState::write(
      std::shared_ptr<FileManager const> fileManager,
      double simTime,
      bool verifyWritesFlag) const {
   std::string filename = generateFilename(std::string("pvp"));
   writeRandState(
         filename,
         fileManager->getMPIBlock(),
         mDataPointer,
         mLayerLoc,
         mExtendedFlag,
         simTime,
         verifyWritesFlag);
}

void CheckpointEntryRandState::read(
      std::shared_ptr<FileManager const> fileManager, double *simTimePtr) const {
   std::string filename = generateFilename(std::string("pvp"));
   *simTimePtr          = readRandState(
         filename,
         fileManager->getMPIBlock(),
         mDataPointer,
         mLayerLoc,
         mExtendedFlag);
}

void CheckpointEntryRandState::remove(std::shared_ptr<FileManager const> fileManager) const {
   deleteFile(fileManager, std::string("pvp"));
}

} // namespace PV
