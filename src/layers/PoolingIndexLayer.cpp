/*
 * PoolingIndexLayer.cpp
 *
 *  Created on: Dec 21, 2010
 *      Author: pschultz
 */

#include "PoolingIndexLayer.hpp"
#include "components/HyPerActivityBuffer.hpp"
#include "components/HyPerActivityComponent.hpp"
#include "components/HyPerInternalStateBuffer.hpp"
#include "components/PoolingIndexLayerInputBuffer.hpp"
#include "components/SingleChannelGSynAccumulator.hpp"

namespace PV {

PoolingIndexLayer::PoolingIndexLayer(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   initialize(paramsIO, comm);
}

PoolingIndexLayer::PoolingIndexLayer() {}

PoolingIndexLayer::~PoolingIndexLayer() {}

void PoolingIndexLayer::initialize(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   HyPerLayer::initialize(paramsIO, comm);
   // This layer is storing its buffers as ints. This is a check to make sure the sizes are the same
   assert(sizeof(int) == sizeof(float));
}

int PoolingIndexLayer::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   int status = HyPerLayer::ioParamsFillGroup(ioSwitch);
   return status;
}

LayerInputBuffer *PoolingIndexLayer::createLayerInput() {
   return new PoolingIndexLayerInputBuffer(mParamsIO, mCommunicator);
}

ActivityComponent *PoolingIndexLayer::createActivityComponent() {
   return new HyPerActivityComponent<SingleChannelGSynAccumulator,
                                     HyPerInternalStateBuffer,
                                     HyPerActivityBuffer>(mParamsIO, mCommunicator);
}

} // end namespace PV
