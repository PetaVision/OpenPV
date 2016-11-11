/*
 * CheckpointEntryPvp.hpp
 *
 *  Created on Sep 27, 2016
 *      Author: Pete Schultz
 */

#ifndef CHECKPOINTENTRYPVP_HPP_
#define CHECKPOINTENTRYPVP_HPP_

#include "CheckpointEntry.hpp"
#include <string>

namespace PV {

template <typename T>
class CheckpointEntryPvp : public CheckpointEntry {
  public:
   CheckpointEntryPvp(
         std::string const &name,
         Communicator *communicator,
         T *dataPtr,
         PVLayerLoc const *layerLoc,
         bool extended)
         : CheckpointEntry(name, communicator),
           mDataPointer(dataPtr),
           mLayerLoc(layerLoc),
           mExtended(extended) {}
   CheckpointEntryPvp(
         std::string const &objName,
         std::string const &dataName,
         Communicator *communicator,
         T *dataPtr,
         PVLayerLoc const *layerLoc,
         bool extended)
         : CheckpointEntry(objName, dataName, communicator),
           mDataPointer(dataPtr),
           mLayerLoc(layerLoc),
           mExtended(extended) {}
   virtual void write(std::string const &checkpointDirectory, double simTime, bool verifyWritesFlag)
         const override;
   virtual void read(std::string const &checkpointDirectory, double *simTimePtr) const override;
   virtual void remove(std::string const &checkpointDirectory) const override;

  private:
   T *calcBatchElementStart(int batchElement) const;

  private:
   T *mDataPointer;
   PVLayerLoc const *mLayerLoc = nullptr;
   bool mExtended              = false;
};

} // end namespace PV

#include "CheckpointEntryPvp.tpp"

#endif // CHECKPOINTENTRYPVP_HPP_