/*
 * StrengthParam.cpp
 *
 *  Created on: Jan 29, 2018
 *      Author: Pete Schultz
 */

#include "StrengthParam.hpp"
#include "connections/BaseConnection.hpp"

namespace PV {

StrengthParam::StrengthParam(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

StrengthParam::~StrengthParam() {}

void StrengthParam::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   BaseObject::initialize(paramsIO, comm);
}

void StrengthParam::setObjectType() { mObjectType = "StrengthParam"; }

int StrengthParam::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   ioParam_strength(ioSwitch);
   return PV_SUCCESS;
}

void StrengthParam::ioParam_strength(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "strength", &mStrength);
}

StrengthParam *StrengthParam::ensureExists(
         std::shared_ptr<CommunicateInitInfoMessage const> message,
         std::shared_ptr<ParamsIO> paramsIO,
         Communicator const *comm) {
   Response::Status status    = Response::NO_ACTION;
   auto objectTable           = message->mObjectTable;
   BaseConnection *parentConn = objectTable->findObject<BaseConnection>(paramsIO->getName());
   FatalIf(                                                                                      
         parentConn == nullptr,    
         "StrengthParam::create() could not find a connection named \"%s\".\n",
         paramsIO->getName().c_str());
   auto *strengthParam = parentConn->getComponentByType<StrengthParam>();
   if (strengthParam) {
      return strengthParam;
   }
   else {                                  
      strengthParam = new StrengthParam(paramsIO, comm);
      parentConn->addUniqueComponent(strengthParam);
   }
   return strengthParam;
}

} // namespace PV
