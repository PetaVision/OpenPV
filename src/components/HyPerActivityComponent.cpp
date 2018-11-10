/*
 * HyPerActivityComponent.cpp
 *
 *  Created on: Oct 12, 2018
 *      Author: Pete Schultz
 */

#include "HyPerActivityComponent.hpp"
#include "components/HyPerActivityBuffer.hpp"
#include "components/HyPerInternalStateBuffer.hpp"
#include "observerpattern/ObserverTable.hpp"

namespace PV {

HyPerActivityComponent::HyPerActivityComponent(
      char const *name,
      PVParams *params,
      Communicator *comm) {
   initialize(name, params, comm);
}

HyPerActivityComponent::~HyPerActivityComponent() {}

void HyPerActivityComponent::initialize(char const *name, PVParams *params, Communicator *comm) {
   ActivityComponent::initialize(name, params, comm);
}

void HyPerActivityComponent::setObjectType() { mObjectType = "HyPerActivityComponent"; }

void HyPerActivityComponent::createComponentTable(char const *tableDescription) {
   ActivityComponent::createComponentTable(tableDescription); // creates Activity
   mInternalState = createInternalState();
   if (mInternalState) {
      addUniqueComponent(mInternalState->getDescription(), mInternalState);
   }
}

ActivityBuffer *HyPerActivityComponent::createActivity() {
   return new HyPerActivityBuffer(getName(), parameters(), mCommunicator);
}

InternalStateBuffer *HyPerActivityComponent::createInternalState() {
   return new HyPerInternalStateBuffer(getName(), parameters(), mCommunicator);
}

Response::Status
HyPerActivityComponent::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   if (mInternalState) {
      mInternalState->respond(message);
   }
   mActivity->respond(message);
   return Response::SUCCESS;
}

Response::Status HyPerActivityComponent::updateActivity(double simTime, double deltaTime) {
   if (mInternalState) {
      mInternalState->updateBuffer(simTime, deltaTime);
   }
   mActivity->updateBuffer(simTime, deltaTime);
   return Response::SUCCESS;
}

} // namespace PV
