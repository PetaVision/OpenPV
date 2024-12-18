/*
 * CheckpointEntryLayerBuffer.hpp
 *
 *  Created on Sep 27, 2016
 *      Author: Pete Schultz
 */

#ifndef CHECKPOINTENTRYLAYERBUFFER_HPP_
#define CHECKPOINTENTRYLAYERBUFFER_HPP_

#include "CheckpointEntryLayerPvp.hpp"
#include "include/PVLayerLoc.hpp"
#include <string>

namespace PV {

template <typename T>
class CheckpointEntryLayerBuffer : public CheckpointEntryLayerPvp<T> {
  public:
   CheckpointEntryLayerBuffer(
         std::string const &name,
         T *dataPtr,
         PVLayerLoc const *layerLoc,
         bool extended)
         : CheckpointEntryLayerPvp<T>(name, layerLoc, extended), mDataPointer(dataPtr) {}
   CheckpointEntryLayerBuffer(
         std::string const &objName,
         std::string const &dataName,
         T *dataPtr,
         PVLayerLoc const *layerLoc,
         bool extended)
         : CheckpointEntryLayerPvp<T>(objName, dataName, layerLoc, extended),
           mDataPointer(dataPtr) {}

  protected:
   virtual int getNumIndices() const override;
   virtual T *calcBatchElementStart(int batchElement, int index) const override;

   T *getDataPointer() const { return mDataPointer; }

  private:
   T *mDataPointer = nullptr;
};

} // end namespace PV

#include "CheckpointEntryLayerBuffer.tpp"

#endif // CHECKPOINTENTRYLAYERBUFFER_HPP_
