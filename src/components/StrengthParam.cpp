/*
 * StrengthParam.cpp
 *
 *  Created on: Jan 29, 2018
 *      Author: Pete Schultz
 */

#include "StrengthParam.hpp"
#include "connections/BaseConnection.hpp"

namespace PV {

StrengthParam::StrengthParam(char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

StrengthParam::~StrengthParam() {}

void StrengthParam::initialize(char const *name, PVParams *params, Communicator const *comm) {
   BaseObject::initialize(name, params, comm);
}

void StrengthParam::setObjectType() { mObjectType = "StrengthParam"; }

int StrengthParam::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_strength(ioFlag);
   return PV_SUCCESS;
}

void StrengthParam::ioParam_strength(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, getName(), "strength", &mStrength, mStrength);
}

StrengthParam *StrengthParam::ensureExists(
         std::shared_ptr<CommunicateInitInfoMessage const> message,
         char const *name,
         PVParams *params,
         Communicator const *comm) {
   Response::Status status    = Response::NO_ACTION;
   auto objectTable           = message->mObjectTable;
   BaseConnection *parentConn = objectTable->findObject<BaseConnection>(name);
   FatalIf(                                                                                      
         parentConn == nullptr,    
         "StrengthParam::create() could not find a connection named \"%s\".\n",
         name);
   auto *strengthParam = parentConn->getComponentByType<StrengthParam>();
   if (strengthParam) {
      return strengthParam;
   }
   else {                                  
      strengthParam = new StrengthParam(name, params, comm);
      parentConn->addUniqueComponent(strengthParam);
   }
   return strengthParam;
}

} // namespace PV
