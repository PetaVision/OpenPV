/*
 * OriginalConnNameParam.cpp
 *
 *  Created on: Jan 5, 2018
 *      Author: Pete Schultz
 */

#include "OriginalConnNameParam.hpp"
#include "observerpattern/ObserverTable.hpp"

namespace PV {

OriginalConnNameParam::OriginalConnNameParam(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

OriginalConnNameParam::~OriginalConnNameParam() {}

void OriginalConnNameParam::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   LinkedObjectParam::initialize(name, params, comm, std::string("originalConnName"));
}

void OriginalConnNameParam::setObjectType() { mObjectType = "OriginalConnNameParam"; }

} // namespace PV
