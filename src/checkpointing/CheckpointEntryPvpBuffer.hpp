/*
 * CheckpointEntryPvpBuffer.hpp
 *
 *  Created on Sep 27, 2016
 *      Author: Pete Schultz
 */

#ifndef CHECKPOINTENTRYPVPBUFFER_HPP_
#define CHECKPOINTENTRYPVPBUFFER_HPP_

#include "CheckpointEntryPvp.hpp"
#include "include/PVLayerLoc.hpp"
#include <string>

namespace PV {

template <typename T>
class CheckpointEntryPvpBuffer : public CheckpointEntryPvp<T> {
  public:
   CheckpointEntryPvpBuffer(
         std::string const &name,
         T *dataPtr,
         PVLayerLoc const *layerLoc,
         bool extended)
         : CheckpointEntryPvp<T>(name, layerLoc, extended), mDataPointer(dataPtr) {}
   CheckpointEntryPvpBuffer(
         std::string const &objName,
         std::string const &dataName,
         T *dataPtr,
         PVLayerLoc const *layerLoc,
         bool extended)
         : CheckpointEntryPvp<T>(objName, dataName, layerLoc, extended),
           mDataPointer(dataPtr) {}

  protected:
   virtual int getNumIndices() const override;
   virtual T *calcBatchElementStart(int batchElement, int index) const override;

   T *getDataPointer() const { return mDataPointer; }

  private:
   T *mDataPointer = nullptr;
};

} // end namespace PV

#include "CheckpointEntryPvpBuffer.tpp"

#endif // CHECKPOINTENTRYPVPBUFFER_HPP_
