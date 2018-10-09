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

ComponentBasedObject *LinkedObjectParam::findLinkedObject(ObserverTable const *hierarchy) {
   ObserverTable *tableComponent = hierarchy->lookupByType<ObserverTable>();
   FatalIf(
         tableComponent == nullptr,
         "%s: CommunicateInitInfoMessage has no ObserverTable.\n",
         getDescription_c());
   std::string linkedName(mLinkedObjectName);
   auto *originalObject = tableComponent->lookupByName<ComponentBasedObject>(linkedName);
   if (originalObject == nullptr) {
      std::string invArgMessage(mParamName);
      invArgMessage.append("No object named \"").append(mLinkedObjectName).append(" \"");
      invArgMessage.append(" in the hierarchy");
      throw std::invalid_argument(invArgMessage);
   }
   return originalObject;
}

} // namespace PV
