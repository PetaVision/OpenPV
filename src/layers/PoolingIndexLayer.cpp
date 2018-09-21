/*
 * PoolingIndexLayer.cpp
 *
 *  Created on: Dec 21, 2010
 *      Author: pschultz
 */

#include "PoolingIndexLayer.hpp"
#include "components/PoolingIndexLayerInputBuffer.hpp"

namespace PV {

PoolingIndexLayer::PoolingIndexLayer() { initialize_base(); }

PoolingIndexLayer::PoolingIndexLayer(const char *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

PoolingIndexLayer::~PoolingIndexLayer() {}

int PoolingIndexLayer::initialize_base() { return PV_SUCCESS; }

int PoolingIndexLayer::initialize(const char *name, HyPerCol *hc) {
   int status = HyPerLayer::initialize(name, hc);
   // This layer is storing its buffers as ints. This is a check to make sure the sizes are the same
   assert(sizeof(int) == sizeof(float));
   assert(status == PV_SUCCESS);
   return status;
}

int PoolingIndexLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerLayer::ioParamsFillGroup(ioFlag);
   return status;
}

LayerInputBuffer *PoolingIndexLayer::createLayerInput() {
   return new PoolingIndexLayerInputBuffer(name, parent);
}

} // end namespace PV
