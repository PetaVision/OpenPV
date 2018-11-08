/*
 * BackgroundActivityComponent.cpp
 *
 *  Created on: 4/16/15
 *  slundquist
 */

#include "BackgroundActivityComponent.hpp"
#include "columns/HyPerCol.hpp"
#include "components/BackgroundActivityUpdater.hpp"

namespace PV {

BackgroundActivityComponent::BackgroundActivityComponent(
      char const *name,
      PVParams *params,
      Communicator *comm) {
   initialize(name, params, comm);
}

BackgroundActivityComponent::~BackgroundActivityComponent() {}

void BackgroundActivityComponent::initialize(
      char const *name,
      PVParams *params,
      Communicator *comm) {
   BaseObject::initialize(name, params, comm);
}

void BackgroundActivityComponent::setObjectType() { mObjectType = "BackgroundActivityComponent"; }

InternalStateBuffer *BackgroundActivityComponent::createInternalStateBuffer() { return nullptr; }

InternalStateUpdater *BackgroundActivityComponent::createInternalStateUpdater() { return nullptr; }

ActivityUpdater *BackgroundActivityComponent::createActivityUpdater() {
   return new BackgroundActivityUpdater(name, parameters(), mCommunicator);
}

} // namespace PV
