/*
 * CheckpointEntryPvpBuffer.hpp
 *
 *  Created on Sep 27, 2016
 *      Author: Pete Schultz
 */

#ifndef CHECKPOINTENTRYPVPBUFFER_HPP_
#define CHECKPOINTENTRYPVPBUFFER_HPP_

#include "CheckpointEntryPvp.hpp"
#include "include/PVLayerLoc.h"
#include <string>

namespace PV {

template <typename T>
class CheckpointEntryPvpBuffer : public CheckpointEntryPvp<T> {
  public:
   CheckpointEntryPvpBuffer(
         std::string const &name,
         std::shared_ptr<MPIBlock const> mpiBlock,
         T *dataPtr,
         PVLayerLoc const *layerLoc,
         bool extended)
         : CheckpointEntryPvp<T>(name, mpiBlock, layerLoc, extended), mDataPointer(dataPtr) {}
   CheckpointEntryPvpBuffer(
         std::string const &objName,
         std::string const &dataName,
         std::shared_ptr<MPIBlock const> mpiBlock,
         T *dataPtr,
         PVLayerLoc const *layerLoc,
         bool extended)
         : CheckpointEntryPvp<T>(objName, dataName, mpiBlock, layerLoc, extended),
           mDataPointer(dataPtr) {}

  protected:
   virtual int getNumFrames() const override;
   virtual T *calcBatchElementStart(int batchElement) const override;
   virtual int calcMPIBatchIndex(int frame) const override;

   T *getDataPointer() const { return mDataPointer; }

  private:
   T *mDataPointer = nullptr;
};

} // end namespace PV

#include "CheckpointEntryPvpBuffer.tpp"

#endif // CHECKPOINTENTRYPVPBUFFER_HPP_
