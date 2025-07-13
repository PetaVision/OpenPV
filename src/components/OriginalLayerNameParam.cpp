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
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

OriginalLayerNameParam::~OriginalLayerNameParam() {}

void OriginalLayerNameParam::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   LinkedObjectParam::initialize(params, defaults, comm, std::string("originalLayerName"));
}

void OriginalLayerNameParam::setObjectType() { mObjectType = "OriginalLayerNameParam"; }

} // namespace PV
