/*
 * HyPerLCALayer.cpp
 *
 *  Created on: Jan 24, 2013
 *      Author: garkenyon
 */

#include "HyPerLCALayer.hpp"
#include "components/ANNActivityBuffer.hpp"
#include "components/GSynAccumulator.hpp"
#include "components/HyPerActivityComponent.hpp"
#include "components/HyPerLCAInternalStateBuffer.hpp"
#include "components/LayerInputBuffer.hpp"

namespace PV {

HyPerLCALayer::HyPerLCALayer(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

HyPerLCALayer::~HyPerLCALayer() {}

void HyPerLCALayer::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   HyPerLayer::initialize(params, defaults, comm);
}

LayerInputBuffer *HyPerLCALayer::createLayerInput() {
   return new LayerInputBuffer(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

ActivityComponent *HyPerLCALayer::createActivityComponent() {
   return new HyPerActivityComponent<GSynAccumulator,
                                     HyPerLCAInternalStateBuffer,
                                     ANNActivityBuffer>(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

} // end namespace PV
