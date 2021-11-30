/*
 * CheckpointEntryPvp.hpp
 *
 *  Created on Feb 13, 2017
 *      Author: Pete Schultz
 */

#ifndef CHECKPOINTENTRYPVP_HPP_
#define CHECKPOINTENTRYPVP_HPP_

#include "CheckpointEntry.hpp"
#include "include/PVLayerLoc.h"
#include <string>
#include <vector>

namespace PV {

template <typename T>
class CheckpointEntryPvp : public CheckpointEntry {
  public:
   CheckpointEntryPvp(
         std::string const &name,
         std::shared_ptr<MPIBlock const> mpiBlock,
         PVLayerLoc const *layerLoc,
         bool extended);
   CheckpointEntryPvp(
         std::string const &objName,
         std::string const &dataName,
         std::shared_ptr<MPIBlock const> mpiBlock,
         PVLayerLoc const *layerLoc,
         bool extended);
   virtual void write(std::string const &checkpointDirectory, double simTime, bool verifyWritesFlag)
         const override;
   virtual void read(std::string const &checkpointDirectory, double *simTimePtr) const override;
   virtual void remove(std::string const &checkpointDirectory) const override;

  protected:
   void initialize(PVLayerLoc const *layerLoc, bool extended);

   virtual int getNumFrames() const                         = 0;
   virtual T *calcBatchElementStart(int batchElement) const = 0;
   virtual int calcMPIBatchIndex(int frame) const           = 0;
   virtual void applyTimestamps(std::vector<double> const &timestamps) const {}

   T *getDataPointer() const { return mDataPointer; }
   PVLayerLoc const *getLayerLoc() const { return mLayerLoc; }
   int getXMargins() const { return mXMargins; }
   int getYMargins() const { return mYMargins; }

  private:
   T *mDataPointer             = nullptr;
   PVLayerLoc const *mLayerLoc = nullptr;

   // If extended is true (reading/writing an extended buffer), use mLayerLoc->halo for the margins
   // If extended is false (reading/writing a restricted buffer), use 0 for the margins.
   int mXMargins = 0;
   int mYMargins = 0;
};

} // end namespace PV

#include "CheckpointEntryPvp.tpp"

#endif // CHECKPOINTENTRYPVP_HPP_
