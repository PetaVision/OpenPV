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
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

OriginalConnNameParam::~OriginalConnNameParam() {}

void OriginalConnNameParam::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   LinkedObjectParam::initialize(params, defaults, comm, std::string("originalConnName"));
}

void OriginalConnNameParam::setObjectType() { mObjectType = "OriginalConnNameParam"; }

} // namespace PV
