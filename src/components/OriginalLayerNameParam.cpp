/*
 * OriginalLayerNameParam.cpp
 *
 *  Created on: Jan 5, 2018
 *      Author: Pete Schultz
 */

#include "OriginalLayerNameParam.hpp"
#include "observerpattern/ObserverTable.hpp"

namespace PV {

OriginalLayerNameParam::OriginalLayerNameParam(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

OriginalLayerNameParam::~OriginalLayerNameParam() {}

void OriginalLayerNameParam::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   LinkedObjectParam::initialize(name, params, comm, std::string("originalLayerName"));
}

void OriginalLayerNameParam::setObjectType() { mObjectType = "OriginalLayerNameParam"; }

} // namespace PV
