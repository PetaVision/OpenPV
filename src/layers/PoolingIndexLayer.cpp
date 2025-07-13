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

PoolingIndexLayer::PoolingIndexLayer(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

PoolingIndexLayer::PoolingIndexLayer() {}

PoolingIndexLayer::~PoolingIndexLayer() {}

void PoolingIndexLayer::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   HyPerLayer::initialize(params, defaults, comm);
   // This layer is storing its buffers as ints. This is a check to make sure the sizes are the same
   assert(sizeof(int) == sizeof(float));
}

int PoolingIndexLayer::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   int status = HyPerLayer::ioParamsFillGroup(ioSwitch);
   return status;
}

LayerInputBuffer *PoolingIndexLayer::createLayerInput() {
   return new PoolingIndexLayerInputBuffer(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

ActivityComponent *PoolingIndexLayer::createActivityComponent() {
   return new HyPerActivityComponent<SingleChannelGSynAccumulator,
                                     HyPerInternalStateBuffer,
                                     HyPerActivityBuffer>(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

} // end namespace PV
