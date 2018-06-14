/*
 * OriginalConnNameParam.cpp
 *
 *  Created on: Jan 5, 2018
 *      Author: Pete Schultz
 */

#include "OriginalConnNameParam.hpp"
#include "columns/HyPerCol.hpp"
#include "columns/ObjectMapComponent.hpp"
#include "components/ConnectionData.hpp"
#include "utils/MapLookupByType.hpp"

namespace PV {

OriginalConnNameParam::OriginalConnNameParam(char const *name, HyPerCol *hc) {
   initialize(name, hc);
}

OriginalConnNameParam::~OriginalConnNameParam() {}

int OriginalConnNameParam::initialize(char const *name, HyPerCol *hc) {
   return BaseObject::initialize(name, hc);
}

void OriginalConnNameParam::setObjectType() { mObjectType = "OriginalConnNameParam"; }

int OriginalConnNameParam::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_originalConnName(ioFlag);
   return PV_SUCCESS;
}

void OriginalConnNameParam::ioParam_originalConnName(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamStringRequired(
         ioFlag, name, "originalConnName", &mOriginalConnName);
}

ComponentBasedObject *
OriginalConnNameParam::findOriginalObject(std::map<std::string, Observer *> const &hierarchy) {
   ObjectMapComponent *objectMapComponent = mapLookupByType<ObjectMapComponent>(hierarchy);
   FatalIf(
         objectMapComponent == nullptr,
         "%s: CommunicateInitInfoMessage has no ObjectMapComponent.\n",
         getDescription_c());
   ComponentBasedObject *originalObject = nullptr;
   originalObject =
         objectMapComponent->lookup<ComponentBasedObject>(std::string(mOriginalConnName));
   if (originalObject == nullptr) {
      std::string invArgMessage("originalConnName \"");
      invArgMessage.append(mOriginalConnName);
      invArgMessage.append("\" does not correspond to an object in the column.");
      throw std::invalid_argument(invArgMessage);
   }
   return originalObject;
}

} // namespace PV
