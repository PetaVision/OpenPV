/*
 * CheckpointEntryDataStore.hpp
 *
 *  Created on Oct 13, 2016
 *      Author: Pete Schultz
 */

#ifndef CHECKPOINTENTRYDATASTORE_HPP_
#define CHECKPOINTENTRYDATASTORE_HPP_

#include <string>

#include "checkpointing/CheckpointEntryLayerPvp.hpp"
#include "structures/DataStore.hpp"

namespace PV {

class CheckpointEntryDataStore : public CheckpointEntryLayerPvp<float> {
  public:
   CheckpointEntryDataStore(
         std::string const &name,
         DataStore *dataStore,
         PVLayerLoc const *layerLoc,
         bool broadcastFlag)
         : CheckpointEntryLayerPvp<float>(
               name, layerLoc, broadcastFlag, true /*extendedFlag*/, dataStore->isSparse()),
           mDataStore(dataStore) {}
   CheckpointEntryDataStore(
         std::string const &objName,
         std::string const &dataName,
         DataStore *dataStore,
         PVLayerLoc const *layerLoc,
         bool broadcastFlag)
         : CheckpointEntryLayerPvp<float>(
               objName,
               dataName,
               layerLoc,
               broadcastFlag,
               true /*extendedFlag*/,
               dataStore->isSparse()),
           mDataStore(dataStore) {}

  protected:
   virtual int getNumIndices() const override;
   virtual float *calcBatchElementStart(int batchElement, int index) const override;
   virtual void applyTimestamps(std::vector<double> const &timestamps) const override {
      setLastUpdateTimes(timestamps);
   }

   virtual void writeSparseBroadcast(
         std::shared_ptr<FileManager const> fileManager,
         std::string const &filename,
         double simTime,
         bool verifyWritesFlag) const override;

   virtual void writeSparseFile(
         std::shared_ptr<FileManager const> fileManager,
         std::string const &filename,
         double simTime,
         bool verifyWritesFlag) const override;

   virtual void loadBatchElement(
         SparseList<float> const &sparseList, int batchElement, int index) const override;

   DataStore *getDataStore() const { return mDataStore; }

  private:
   void setLastUpdateTimes(std::vector<double> const &timestamps) const;

  private:
   DataStore *mDataStore = nullptr;
};

} // end namespace PV

#endif // CHECKPOINTENTRYDATASTORE_HPP_
