/*
 * LinkedObjectParam.cpp
 *
 *  Created on: Jun 8, 2018
 *      Author: Pete Schultz
 */

#include "LinkedObjectParam.hpp"
#include "observerpattern/ObserverTable.hpp"

namespace PV {

LinkedObjectParam::~LinkedObjectParam() {}

void LinkedObjectParam::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm,
      std::string const &paramName) {
   mParamName = paramName;
   BaseObject::initialize(paramsIO, comm);
}

int LinkedObjectParam::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   ioParam_linkedObjectName(ioSwitch);
   return PV_SUCCESS;
}

void LinkedObjectParam::ioParam_linkedObjectName(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, mParamName.c_str(), &mLinkedObjectName);
}

} // namespace PV
