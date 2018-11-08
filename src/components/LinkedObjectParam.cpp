/*
 * LinkedObjectParam.cpp
 *
 *  Created on: Jun 8, 2018
 *      Author: Pete Schultz
 */

#include "LinkedObjectParam.hpp"
#include "columns/HyPerCol.hpp"
#include "observerpattern/ObserverTable.hpp"

namespace PV {

LinkedObjectParam::~LinkedObjectParam() {}

void LinkedObjectParam::initialize(
      char const *name,
      PVParams *params,
      Communicator *comm,
      std::string const &paramName) {
   mParamName = paramName;
   BaseObject::initialize(name, params, comm);
}

int LinkedObjectParam::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_linkedObjectName(ioFlag);
   return PV_SUCCESS;
}

void LinkedObjectParam::ioParam_linkedObjectName(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamStringRequired(ioFlag, name, mParamName.c_str(), &mLinkedObjectName);
}

ComponentBasedObject *LinkedObjectParam::findLinkedObject(ObserverTable const *hierarchy) {
   ObserverTable const *tableComponent = hierarchy;
   std::string linkedName(mLinkedObjectName);
   int maxIterations = 2; // Limits the depth of the recursion when searching for dependencies.
   auto *originalObject =
         tableComponent->lookupByNameRecursive<ComponentBasedObject>(linkedName, maxIterations);
   if (originalObject == nullptr) {
      std::string invArgMessage(mParamName);
      invArgMessage.append("No object named \"").append(mLinkedObjectName).append(" \"");
      invArgMessage.append(" in the hierarchy");
      throw std::invalid_argument(invArgMessage);
   }
   return originalObject;
}

} // namespace PV
