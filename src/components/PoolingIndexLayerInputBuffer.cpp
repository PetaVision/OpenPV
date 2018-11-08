/*
 * PoolingIndexLayerInputBuffer.cpp
 *
 *  Created on: Sep 18, 2018
 *      Author: Pete Schultz
 */

#include "PoolingIndexLayerInputBuffer.hpp"
#include "columns/HyPerCol.hpp"

namespace PV {

PoolingIndexLayerInputBuffer::PoolingIndexLayerInputBuffer(
      char const *name,
      PVParams *params,
      Communicator *comm) {
   initialize(name, params, comm);
}

PoolingIndexLayerInputBuffer::~PoolingIndexLayerInputBuffer() {}

void PoolingIndexLayerInputBuffer::initialize(
      char const *name,
      PVParams *params,
      Communicator *comm) {
   LayerInputBuffer::initialize(name, params, comm);
}

void PoolingIndexLayerInputBuffer::setObjectType() { mObjectType = "PoolingIndexLayerInputBuffer"; }

void PoolingIndexLayerInputBuffer::resetGSynBuffers(double simulationTime, double deltaTime) {
   // Reset GSynBuffers does nothing, as the orig pooling connection deals with clearing this buffer
}

} // namespace PV
