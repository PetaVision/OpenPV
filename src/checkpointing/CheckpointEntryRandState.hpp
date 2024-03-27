/*
 * CheckpointEntryRandState.hpp
 *
 *  Created on Oct 6, 2016
 *      Author: Pete Schultz
 */

#ifndef CHECKPOINTENTRYRANDSTATE_HPP_
#define CHECKPOINTENTRYRANDSTATE_HPP_

#include "CheckpointEntry.hpp"
#include "include/PVLayerLoc.hpp"
#include "utils/cl_random.h"

namespace PV {

class CheckpointEntryRandState : public CheckpointEntry {
  public:
   CheckpointEntryRandState(
         std::string const &dataName,
         taus_uint4 *dataPointer,
         PVLayerLoc *layerLoc,
         bool extendedFlag)
         : CheckpointEntry(dataName),
           mDataPointer(dataPointer),
           mLayerLoc(layerLoc),
           mExtendedFlag(extendedFlag) {}
   CheckpointEntryRandState(
         std::string const &objName,
         std::string const &dataName,
         taus_uint4 *dataPointer,
         PVLayerLoc const *layerLoc,
         bool extendedFlag)
         : CheckpointEntry(objName, dataName),
           mDataPointer(dataPointer),
           mLayerLoc(layerLoc),
           mExtendedFlag(extendedFlag) {}
   virtual void
   write(std::shared_ptr<FileManager const> fileManager, double simTime, bool verifyWritesFlag)
         const override;
   virtual void
   read(std::shared_ptr<FileManager const> fileManager, double *simTimePtr) const override;
   virtual void remove(std::shared_ptr<FileManager const> fileManager) const override;

  private:
   taus_uint4 *mDataPointer;
   PVLayerLoc const *mLayerLoc;
   bool mExtendedFlag;
};

} // namespace PV

#endif // CHECKPOINTENTRYRANDSTATE_HPP_
