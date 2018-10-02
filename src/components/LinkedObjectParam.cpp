/*
 * LinkedObjectParam.cpp
 *
 *  Created on: Jun 8, 2018
 *      Author: Pete Schultz
 */

#include "LinkedObjectParam.hpp"
#include "columns/HyPerCol.hpp"
#include "columns/ObserverTableComponent.hpp"
#include "observerpattern/ObserverTable.hpp"

namespace PV {

LinkedObjectParam::~LinkedObjectParam() {}

int LinkedObjectParam::initialize(char const *name, HyPerCol *hc, std::string const &paramName) {
   mParamName = paramName;
   return BaseObject::initialize(name, hc);
}

int LinkedObjectParam::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_linkedObjectName(ioFlag);
   return PV_SUCCESS;
}

void LinkedObjectParam::ioParam_linkedObjectName(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamStringRequired(ioFlag, name, mParamName.c_str(), &mLinkedObjectName);
}

ComponentBasedObject *LinkedObjectParam::findLinkedObject(ObserverTable const &hierarchy) {
   ObserverTableComponent *tableComponent = hierarchy.lookupByType<ObserverTableComponent>();
   FatalIf(
         tableComponent == nullptr,
         "%s: CommunicateInitInfoMessage has no ObserverTableComponent.\n",
         getDescription_c());
   auto &observerTable = tableComponent->getObserverTable();
   ComponentBasedObject *originalObject =
         observerTable.lookup<ComponentBasedObject>(std::string(mLinkedObjectName));
   if (originalObject == nullptr) {
      std::string invArgMessage(mParamName);
      invArgMessage.append(" \"").append(mLinkedObjectName).append(" \"");
      invArgMessage.append(" does not correspond to an object in the column.");
      throw std::invalid_argument(invArgMessage);
   }
   return originalObject;
}

} // namespace PV
