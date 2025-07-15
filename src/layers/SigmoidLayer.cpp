/*
 * SigmoidLayer.cpp
 *
 *  Created on: May 11, 2011
 *      Author: garkenyon
 */

#include "SigmoidLayer.hpp"
#include "components/CloneActivityComponent.hpp"
#include "components/CloneInternalStateBuffer.hpp"
#include "components/SigmoidActivityBuffer.hpp"

// SigmoidLayer can be used to implement Sigmoid junctions
namespace PV {
SigmoidLayer::SigmoidLayer(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   initialize(paramsIO, comm);
}

SigmoidLayer::SigmoidLayer() {}

SigmoidLayer::~SigmoidLayer() {}

void SigmoidLayer::initialize(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   CloneVLayer::initialize(paramsIO, comm);
}

ActivityComponent *SigmoidLayer::createActivityComponent() {
   return new CloneActivityComponent<CloneInternalStateBuffer, SigmoidActivityBuffer>(
         mParamsIO, mCommunicator);
}

} // end namespace PV
