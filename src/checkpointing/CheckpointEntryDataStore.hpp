/*
 * CheckpointEntryDataStore.hpp
 *
 *  Created on Oct 13, 2016
 *      Author: Pete Schultz
 */

#ifndef CHECKPOINTENTRYDATASTORE_HPP_
#define CHECKPOINTENTRYDATASTORE_HPP_

#include "checkpointing/CheckpointEntryPvp.hpp"
#include "columns/DataStore.hpp"
#include <string>

namespace PV {

class CheckpointEntryDataStore : public CheckpointEntryPvp<float> {
  public:
   CheckpointEntryDataStore(
         std::string const &name,
         MPIBlock const *mpiBlock,
         DataStore *dataStore,
         PVLayerLoc const *layerLoc)
         : CheckpointEntryPvp<float>(name, mpiBlock, layerLoc, true), mDataStore(dataStore) {}
   CheckpointEntryDataStore(
         std::string const &objName,
         std::string const &dataName,
         MPIBlock const *mpiBlock,
         DataStore *dataStore,
         PVLayerLoc const *layerLoc)
         : CheckpointEntryPvp<float>(objName, dataName, mpiBlock, layerLoc, true),
           mDataStore(dataStore) {}

  protected:
   virtual int getNumFrames() const override;
   virtual float *calcBatchElementStart(int batchElement) const override;
   virtual int calcMPIBatchIndex(int frame) const override;
   virtual void applyTimestamps(std::vector<double> const &timestamps) const override {
      setLastUpdateTimes(timestamps);
   }

   DataStore *getDataStore() const { return mDataStore; }

  private:
   void setLastUpdateTimes(std::vector<double> const &timestamps) const;

  private:
   DataStore *mDataStore = nullptr;
};

} // end namespace PV

#endif // CHECKPOINTENTRYDATASTORE_HPP_
