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
SigmoidLayer::SigmoidLayer(const char *name, PVParams *params, Communicator *comm) {
   initialize(name, params, comm);
}

SigmoidLayer::SigmoidLayer() {}

SigmoidLayer::~SigmoidLayer() {}

void SigmoidLayer::initialize(const char *name, PVParams *params, Communicator *comm) {
   CloneVLayer::initialize(name, params, comm);
}

ActivityComponent *SigmoidLayer::createActivityComponent() {
   return new CloneActivityComponent<CloneInternalStateBuffer, SigmoidActivityBuffer>(
         getName(), parameters(), mCommunicator);
}

} // end namespace PV
