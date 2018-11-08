/*
 * PoolingIndexLayer.cpp
 *
 *  Created on: Dec 21, 2010
 *      Author: pschultz
 */

#include "PoolingIndexLayer.hpp"
#include "components/PoolingIndexLayerInputBuffer.hpp"

namespace PV {

PoolingIndexLayer::PoolingIndexLayer(const char *name, PVParams *params, Communicator *comm) {
   initialize(name, params, comm);
}

PoolingIndexLayer::PoolingIndexLayer() {}

PoolingIndexLayer::~PoolingIndexLayer() {}

void PoolingIndexLayer::initialize(const char *name, PVParams *params, Communicator *comm) {
   HyPerLayer::initialize(name, params, comm);
   // This layer is storing its buffers as ints. This is a check to make sure the sizes are the same
   assert(sizeof(int) == sizeof(float));
}

int PoolingIndexLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerLayer::ioParamsFillGroup(ioFlag);
   return status;
}

LayerInputBuffer *PoolingIndexLayer::createLayerInput() {
   return new PoolingIndexLayerInputBuffer(name, parameters(), mCommunicator);
}

} // end namespace PV
