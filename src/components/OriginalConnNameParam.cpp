/*
 * OriginalConnNameParam.cpp
 *
 *  Created on: Jan 5, 2018
 *      Author: Pete Schultz
 */

#include "OriginalConnNameParam.hpp"
#include "observerpattern/ObserverTable.hpp"

namespace PV {

OriginalConnNameParam::OriginalConnNameParam(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

OriginalConnNameParam::~OriginalConnNameParam() {}

void OriginalConnNameParam::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   LinkedObjectParam::initialize(paramsIO, comm, std::string("originalConnName"));
}

void OriginalConnNameParam::setObjectType() { mObjectType = "OriginalConnNameParam"; }

} // namespace PV
