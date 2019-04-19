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

void LinkedObjectParam::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm,
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

} // namespace PV
