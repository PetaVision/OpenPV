/*
 * CheckpointEntryLayerPvp.hpp
 *
 *  Created on Feb 13, 2017
 *      Author: Pete Schultz
 */

#ifndef CHECKPOINTENTRYLAYERPVP_HPP_
#define CHECKPOINTENTRYLAYERPVP_HPP_

#include "CheckpointEntry.hpp"
#include "structures/PVLayerLoc.hpp"
#include <string>
#include <vector>

namespace PV {

template <typename T>
class CheckpointEntryLayerPvp : public CheckpointEntry {
  public:
   CheckpointEntryLayerPvp(
         std::string const &name,
         PVLayerLoc const *layerLoc,
         bool broadcastFlag,
         bool extendedFlag);
   CheckpointEntryLayerPvp(
         std::string const &objName,
         std::string const &dataName,
         PVLayerLoc const *layerLoc,
         bool broadcastFlag,
         bool extendedFlag);
   virtual void write(
         std::shared_ptr<FileManager const> fileManager, double simTime, bool verifyWritesFlag)
         const override;
   virtual void
         read(std::shared_ptr<FileManager const> fileManager, double *simTimePtr) const override;
   virtual void remove(std::shared_ptr<FileManager const> fileManager) const override;

  protected:
   void initialize(PVLayerLoc const *layerLoc, bool broadcastFlag, bool extendedFlag);

   virtual int getNumIndices() const = 0;

   virtual T *calcBatchElementStart(int batchElement, int index) const = 0;

   /**
    * Sets the array dataStart to all zeros. The size of the array
    * is the local size determined by loc. The extended flag determines
    * whether to use the restricted size or the extended size.
    * This is included for backwards compatibility with the
    * behavior pre-FileManager, but is probably not necessary.
    */
   void clearData(T *dataStart, PVLayerLoc const *loc, bool extended) const;
   virtual void applyTimestamps(std::vector<double> const &timestamps) const {}

   T *getDataPointer() const { return mDataPointer; }
   PVLayerLoc const *getLayerLoc() const { return mLayerLoc; }
   bool getBroadcastFlag() const { return mBroadcastFlag; }
   bool getExtendedFlag() const { return mExtendedFlag; }

  private:
   T *mDataPointer             = nullptr;
   PVLayerLoc const *mLayerLoc = nullptr;

   bool mBroadcastFlag;
   bool mExtendedFlag;
};

} // end namespace PV

#include "CheckpointEntryLayerPvp.tpp"

#endif // CHECKPOINTENTRYLAYERPVP_HPP_
