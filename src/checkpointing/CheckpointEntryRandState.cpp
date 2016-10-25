/*
 * CheckpointEntryRandState.cpp
 *
 *  Created on Oct 6, 2016
 *      Author: Pete Schultz
 */

#include "CheckpointEntryRandState.hpp"
#include "io/fileio.hpp"
#include "structures/Buffer.hpp"
#include "utils/BufferUtilsMPI.hpp"

namespace PV {

void CheckpointEntryRandState::write(
      std::string const &checkpointDirectory,
      double simTime,
      bool verifyWritesFlag) const {
   int nxLocal = mLayerLoc->nx;
   int nyLocal = mLayerLoc->ny;
   if (mExtendedFlag) {
      nxLocal += mLayerLoc->halo.lt + mLayerLoc->halo.rt;
      nyLocal += mLayerLoc->halo.dn + mLayerLoc->halo.up;
   }
   int nf = mLayerLoc->nf;
   Buffer<taus_uint4> localBuffer{mDataPointer, nxLocal, nyLocal, nf};
   Buffer<taus_uint4> globalBuffer = BufferUtils::gather(getCommunicator(), localBuffer, mLayerLoc->nx, mLayerLoc->ny);
   if (getCommunicator()->commRank()==0) {
      std::string path = generatePath(checkpointDirectory, "bin");
      FileStream fileStream{path.c_str(), std::ios_base::out, verifyWritesFlag};
      fileStream.write(globalBuffer.asVector().data(), globalBuffer.getTotalElements()*sizeof(taus_uint4));
   }
}

void CheckpointEntryRandState::read(std::string const &checkpointDirectory, double *simTimePtr)
      const {
   std::string path = generatePath(checkpointDirectory, "bin");
   readRandState(path.c_str(), getCommunicator(), mDataPointer, mLayerLoc, mExtendedFlag);
}

void CheckpointEntryRandState::remove(std::string const &checkpointDirectory) const {
   deleteFile(checkpointDirectory, "bin");
}

} // namespace PV
