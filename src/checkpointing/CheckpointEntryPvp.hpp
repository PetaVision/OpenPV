/*
 * CheckpointEntryPvp.hpp
 *
 *  Created on Feb 13, 2017
 *      Author: Pete Schultz
 */

#ifndef CHECKPOINTENTRYPVP_HPP_
#define CHECKPOINTENTRYPVP_HPP_

#include "CheckpointEntry.hpp"
#include "include/PVLayerLoc.hpp"
#include <string>
#include <vector>

namespace PV {

template <typename T>
class CheckpointEntryPvp : public CheckpointEntry {
  public:
   CheckpointEntryPvp(
         std::string const &name,
         PVLayerLoc const *layerLoc,
         bool extended);
   CheckpointEntryPvp(
         std::string const &objName,
         std::string const &dataName,
         PVLayerLoc const *layerLoc,
         bool extended);
   virtual void write(
         std::shared_ptr<FileManager const> fileManager, double simTime, bool verifyWritesFlag)
         const override;
   virtual void
         read(std::shared_ptr<FileManager const> fileManager, double *simTimePtr) const override;
   virtual void remove(std::shared_ptr<FileManager const> fileManager) const override;

  protected:
   void initialize(PVLayerLoc const *layerLoc, bool extended);

   virtual int getNumIndices() const                                   = 0;
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
   bool getExtended() const { return mExtended; }

  private:
   T *mDataPointer             = nullptr;
   PVLayerLoc const *mLayerLoc = nullptr;

   bool mExtended;
};

} // end namespace PV

#include "CheckpointEntryPvp.tpp"

#endif // CHECKPOINTENTRYPVP_HPP_
