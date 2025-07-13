/*
 * InputLayerNameParam.cpp
 *
 *  Created on: Oct 12, 2018
 *      Author: Pete Schultz
 */

#include "InputLayerNameParam.hpp"
#include "observerpattern/ObserverTable.hpp"

namespace PV {

InputLayerNameParam::InputLayerNameParam(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

InputLayerNameParam::~InputLayerNameParam() {}

void InputLayerNameParam::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   LinkedObjectParam::initialize(params, defaults, comm, std::string("inputLayerName"));
}

void InputLayerNameParam::setObjectType() { mObjectType = "InputLayerNameParam"; }

} // namespace PV
