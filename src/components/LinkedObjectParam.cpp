/*
 * LinkedObjectParam.cpp
 *
 *  Created on: Jun 8, 2018
 *      Author: Pete Schultz
 */

#include "LinkedObjectParam.hpp"
#include "columns/HyPerCol.hpp"
#include "columns/ObjectMapComponent.hpp"
#include "utils/MapLookupByType.hpp"

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
   parent->parameters()->ioParamStringRequired(
         ioFlag, name, mParamName.c_str(), &mLinkedObjectName);
}

ComponentBasedObject *
LinkedObjectParam::findLinkedObject(std::map<std::string, Observer *> const &hierarchy) {
   ObjectMapComponent *objectMapComponent = mapLookupByType<ObjectMapComponent>(hierarchy);
   FatalIf(
         objectMapComponent == nullptr,
         "%s: CommunicateInitInfoMessage has no ObjectMapComponent.\n",
         getDescription_c());
   ComponentBasedObject *originalObject = nullptr;
   originalObject =
         objectMapComponent->lookup<ComponentBasedObject>(std::string(mLinkedObjectName));
   if (originalObject == nullptr) {
      std::string invArgMessage(mParamName);
      invArgMessage.append(" \"").append(mLinkedObjectName).append(" \"");
      invArgMessage.append(" does not correspond to an object in the column.");
      throw std::invalid_argument(invArgMessage);
   }
   return originalObject;
}

} // namespace PV
