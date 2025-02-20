/*
 * CheckpointEntryLayerBuffer.hpp
 *
 *  Created on Sep 27, 2016
 *      Author: Pete Schultz
 */

#ifndef CHECKPOINTENTRYLAYERBUFFER_HPP_
#define CHECKPOINTENTRYLAYERBUFFER_HPP_

#include "CheckpointEntryLayerPvp.hpp"
#include "structures/PVLayerLoc.hpp"
#include <string>

namespace PV {

template <typename T>
class CheckpointEntryLayerBuffer : public CheckpointEntryLayerPvp<T> {
  public:
   CheckpointEntryLayerBuffer(
         std::string const &name,
         T *dataPtr,
         PVLayerLoc const *layerLoc,
         bool broadcastFlag,
         bool extendedFlag)
         : CheckpointEntryLayerPvp<T>(name, layerLoc, broadcastFlag, extendedFlag),
           mDataPointer(dataPtr) {}
   CheckpointEntryLayerBuffer(
         std::string const &objName,
         std::string const &dataName,
         T *dataPtr,
         PVLayerLoc const *layerLoc,
         bool broadcastFlag,
         bool extendedFlag)
         : CheckpointEntryLayerPvp<T>(objName, dataName, layerLoc, broadcastFlag, extendedFlag),
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
