/*
 * CPTestInputLayer.cpp
 */

#include "CPTestInputLayer.hpp"
#include "CPTestInputInternalStateBuffer.hpp"
#include <components/GSynAccumulator.hpp>
#include <components/HyPerActivityBuffer.hpp>
#include <components/HyPerActivityComponent.hpp>

namespace PV {

CPTestInputLayer::CPTestInputLayer(const char *name, PVParams *params, Communicator *comm) {
   initialize(name, params, comm);
}

CPTestInputLayer::~CPTestInputLayer() {}

void CPTestInputLayer::initialize(const char *name, PVParams *params, Communicator *comm) {
   HyPerLayer::initialize(name, params, comm);
}

ActivityComponent *CPTestInputLayer::createActivityComponent() {
   return new HyPerActivityComponent<GSynAccumulator,
                                     CPTestInputInternalStateBuffer,
                                     HyPerActivityBuffer>(getName(), parameters(), mCommunicator);
}

} // end namespace PV
