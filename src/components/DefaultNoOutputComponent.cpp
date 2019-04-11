/*
 * DefaultNoOutputComponent.cpp
 *
 *  Created on: Dec 3, 2018
 *      Author: peteschultz
 */

#include "DefaultNoOutputComponent.hpp"

namespace PV {

DefaultNoOutputComponent::DefaultNoOutputComponent(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

DefaultNoOutputComponent::DefaultNoOutputComponent() {}

DefaultNoOutputComponent::~DefaultNoOutputComponent() {}

void DefaultNoOutputComponent::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   mWriteStep = -1;
   LayerOutputComponent::initialize(name, params, comm);
}

void DefaultNoOutputComponent::setObjectType() { mObjectType = "DefaultNoOutputComponent"; }

} // namespace PV
