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

BackgroundActivityComponent::BackgroundActivityComponent(char const *name, HyPerCol *hc) {
   initialize(name, hc);
}

BackgroundActivityComponent::~BackgroundActivityComponent() {}

int BackgroundActivityComponent::initialize(char const *name, HyPerCol *hc) {
   return BaseObject::initialize(name, hc);
}

void BackgroundActivityComponent::setObjectType() { mObjectType = "BackgroundActivityComponent"; }

InternalStateBuffer *BackgroundActivityComponent::createInternalStateBuffer() { return nullptr; }

InternalStateUpdater *BackgroundActivityComponent::createInternalStateUpdater() { return nullptr; }

ActivityUpdater *BackgroundActivityComponent::createActivityUpdater() {
   return new BackgroundActivityUpdater(name, parent);
}

} // namespace PV
