/*
 * CheckpointEntryDataStore.hpp
 *
 *  Created on Oct 13, 2016
 *      Author: Pete Schultz
 */

#ifndef CHECKPOINTENTRYDATASTORE_HPP_
#define CHECKPOINTENTRYDATASTORE_HPP_

#include "columns/Communicator.hpp"
#include "io/CheckpointEntry.hpp"
#include "structures/RingBuffer.hpp"
#include <string>

namespace PV {

class CheckpointEntryDataStore : public CheckpointEntry {
  public:
   CheckpointEntryDataStore(
         std::string const &name,
         Communicator *communicator,
         RingBuffer<pvdata_t> *buffer,
         RingBuffer<double> *lastUpdateTimes,
         PVLayerLoc const *layerLoc)
         : CheckpointEntry(name, communicator) { initialize(buffer, lastUpdateTimes, layerLoc); }
   CheckpointEntryDataStore(
         std::string const &objName,
         std::string const &dataName,
         Communicator *communicator,
         RingBuffer<pvdata_t> *buffer,
         RingBuffer<double> *lastUpdateTimes,
         PVLayerLoc const *layerLoc)
         : CheckpointEntry(objName, dataName, communicator) { initialize(buffer, lastUpdateTimes, layerLoc); }
   virtual void write(std::string const &checkpointDirectory, double simTime, bool verifyWritesFlag)
         const override;
   virtual void read(std::string const &checkpointDirectory, double *simTimePtr) const override;
   virtual void remove(std::string const &checkpointDirectory) const;

  protected:
   void initialize(RingBuffer<pvdata_t> *buffer, RingBuffer<double> *lastUpdateTimes, PVLayerLoc const *layerLoc);

  private:
   RingBuffer<pvdata_t> *mRingBuffer;
   RingBuffer<double> *mLastUpdateTimes;
   PVLayerLoc const *mLayerLoc = nullptr;
};

} // end namespace PV

#endif // CHECKPOINTENTRYDATASTORE_HPP_
