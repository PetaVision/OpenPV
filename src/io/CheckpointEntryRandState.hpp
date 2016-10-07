/*
 * CheckpointEntryRandState.hpp
 *
 *  Created on Oct 6, 2016
 *      Author: Pete Schultz
 */

#ifndef CHECKPOINTENTRYRANDSTATE_HPP_
#define CHECKPOINTENTRYRANDSTATE_HPP_

#include "CheckpointEntry.hpp"
#include "include/pv_types.h"

namespace PV {

class CheckpointEntryRandState : public CheckpointEntry {
  public:
   CheckpointEntryRandState(
         std::string const &dataName,
         Communicator *communicator,
         taus_uint4 *dataPointer,
         PVLayerLoc *layerLoc,
         bool extendedFlag)
         : CheckpointEntry(dataName, communicator),
           mDataPointer(dataPointer),
           mLayerLoc(layerLoc),
           mExtendedFlag(extendedFlag) {}
   CheckpointEntryRandState(
         std::string const &objName,
         std::string const &dataName,
         Communicator *communicator,
         taus_uint4 *dataPointer,
         PVLayerLoc const *layerLoc,
         bool extendedFlag)
         : CheckpointEntry(objName, dataName, communicator),
           mDataPointer(dataPointer),
           mLayerLoc(layerLoc),
           mExtendedFlag(extendedFlag) {}
   virtual void write(std::string const &checkpointDirectory, double simTime, bool verifyWritesFlag)
         const override;
   virtual void read(std::string const &checkpointDirectory, double *simTimePtr) const override;
   virtual void remove(std::string const &checkpointDirectory) const override;

  private:
   taus_uint4 *mDataPointer;
   PVLayerLoc const *mLayerLoc;
   bool mExtendedFlag;
};

}  // namespace PV

#endif // CHECKPOINTENTRYRANDSTATE_HPP_

