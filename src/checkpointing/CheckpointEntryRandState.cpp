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
      std::string const &checkpointDirectory,
      double simTime,
      bool verifyWritesFlag) const {
   std::string path = generatePath(checkpointDirectory, "bin");
   writeRandState(
         path, getCommunicator(), mDataPointer, mLayerLoc, mExtendedFlag, verifyWritesFlag);
}

void CheckpointEntryRandState::read(std::string const &checkpointDirectory, double *simTimePtr)
      const {
   std::string path = generatePath(checkpointDirectory, "bin");
   readRandState(path, getCommunicator(), mDataPointer, mLayerLoc, mExtendedFlag);
}

void CheckpointEntryRandState::remove(std::string const &checkpointDirectory) const {
   deleteFile(checkpointDirectory, "bin");
}

} // namespace PV
