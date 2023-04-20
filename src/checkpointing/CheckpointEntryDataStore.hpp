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
         DataStore *dataStore,
         PVLayerLoc const *layerLoc)
         : CheckpointEntryPvp<float>(name, layerLoc, true), mDataStore(dataStore) {}
   CheckpointEntryDataStore(
         std::string const &objName,
         std::string const &dataName,
         DataStore *dataStore,
         PVLayerLoc const *layerLoc)
         : CheckpointEntryPvp<float>(objName, dataName, layerLoc, true),
           mDataStore(dataStore) {}

   virtual void
         read(std::shared_ptr<FileManager const> fileManager, double *simTimePtr) const override;

  protected:
   virtual int getNumIndices() const override;
   virtual float *calcBatchElementStart(int batchElement, int index) const override;
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
