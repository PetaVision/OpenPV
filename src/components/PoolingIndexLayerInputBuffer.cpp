/*
 * PoolingIndexLayerInputBuffer.cpp
 *
 *  Created on: Sep 18, 2018
 *      Author: Pete Schultz
 */

#include "PoolingIndexLayerInputBuffer.hpp"
#include "columns/HyPerCol.hpp"

namespace PV {

PoolingIndexLayerInputBuffer::PoolingIndexLayerInputBuffer(char const *name, HyPerCol *hc) {
   initialize(name, hc);
}

PoolingIndexLayerInputBuffer::~PoolingIndexLayerInputBuffer() {}

int PoolingIndexLayerInputBuffer::initialize(char const *name, HyPerCol *hc) {
   int status = LayerInputBuffer::initialize(name, hc);
   return status;
}

void PoolingIndexLayerInputBuffer::setObjectType() { mObjectType = "PoolingIndexLayerInputBuffer"; }

void PoolingIndexLayerInputBuffer::resetGSynBuffers(double simulationTime, double deltaTime) {
   // Reset GSynBuffers does nothing, as the orig pooling connection deals with clearing this buffer
}

} // namespace PV
