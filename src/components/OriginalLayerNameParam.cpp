/*
 * OriginalLayerNameParam.cpp
 *
 *  Created on: Jan 5, 2018
 *      Author: Pete Schultz
 */

#include "OriginalLayerNameParam.hpp"
#include "observerpattern/ObserverTable.hpp"

namespace PV {

OriginalLayerNameParam::OriginalLayerNameParam(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

OriginalLayerNameParam::~OriginalLayerNameParam() {}

void OriginalLayerNameParam::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   LinkedObjectParam::initialize(paramsIO, comm, std::string("originalLayerName"));
}

void OriginalLayerNameParam::setObjectType() { mObjectType = "OriginalLayerNameParam"; }

} // namespace PV
