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

HyPerLCALayer::HyPerLCALayer(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   initialize(paramsIO, comm);
}

HyPerLCALayer::~HyPerLCALayer() {}

void HyPerLCALayer::initialize(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   HyPerLayer::initialize(paramsIO, comm);
}

LayerInputBuffer *HyPerLCALayer::createLayerInput() {
   return new LayerInputBuffer(mParamsIO, mCommunicator);
}

ActivityComponent *HyPerLCALayer::createActivityComponent() {
   return new HyPerActivityComponent<GSynAccumulator,
                                     HyPerLCAInternalStateBuffer,
                                     ANNActivityBuffer>(mParamsIO, mCommunicator);
}

} // end namespace PV
