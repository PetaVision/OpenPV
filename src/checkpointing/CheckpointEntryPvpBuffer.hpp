/*
 * CheckpointEntryPvpBuffer.hpp
 *
 *  Created on Sep 27, 2016
 *      Author: Pete Schultz
 */

#ifndef CHECKPOINTENTRYPVPBUFFER_HPP_
#define CHECKPOINTENTRYPVPBUFFER_HPP_

#include "CheckpointEntry.hpp"
#include "include/PVLayerLoc.h"
#include <string>

namespace PV {

template <typename T>
class CheckpointEntryPvpBuffer : public CheckpointEntry {
  public:
   CheckpointEntryPvpBuffer(
         std::string const &name,
         MPIBlock const *mpiBlock,
         T *dataPtr,
         PVLayerLoc const *layerLoc,
         bool extended);
   CheckpointEntryPvpBuffer(
         std::string const &objName,
         std::string const &dataName,
         MPIBlock const *mpiBlock,
         T *dataPtr,
         PVLayerLoc const *layerLoc,
         bool extended);
   virtual void write(std::string const &checkpointDirectory, double simTime, bool verifyWritesFlag)
         const override;
   virtual void read(std::string const &checkpointDirectory, double *simTimePtr) const override;
   virtual void remove(std::string const &checkpointDirectory) const override;

  protected:
   void initialize(T *dataPtr, PVLayerLoc const *layerLoc, bool extended);

  private:
   int getNumFrames() const;
   T *calcBatchElementStart(int batchElement) const;
   int calcMPIBatchIndex(int frame) const;

  private:
   T *mDataPointer             = nullptr;
   PVLayerLoc const *mLayerLoc = nullptr;
   int mXMargins               = 0; // If extended is true, use mLayerLoc's halo for the margins.
   int mYMargins               = 0; // If extended is false, use zero for the margins.
};

} // end namespace PV

#include "CheckpointEntryPvpBuffer.tpp"

#endif // CHECKPOINTENTRYPVPBUFFER_HPP_
