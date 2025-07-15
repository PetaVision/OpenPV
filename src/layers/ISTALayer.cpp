/*
 * ISTALayer.cpp
 *
 *  Created on: Jan 24, 2013
 *      Author: garkenyon
 */

#include "ISTALayer.hpp"
#include "components/ANNActivityBuffer.hpp"
#include "components/GSynAccumulator.hpp"
#include "components/HyPerActivityComponent.hpp"
#include "components/ISTAInternalStateBuffer.hpp"
#include "components/LayerInputBuffer.hpp"

namespace PV {

ISTALayer::ISTALayer(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   initialize(paramsIO, comm);
}

ISTALayer::~ISTALayer() {}

void ISTALayer::initialize(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   HyPerLayer::initialize(paramsIO, comm);
}

ActivityComponent *ISTALayer::createActivityComponent() {
   return new HyPerActivityComponent<GSynAccumulator, ISTAInternalStateBuffer, ANNActivityBuffer>(
         mParamsIO, mCommunicator);
}

} // end namespace PV
