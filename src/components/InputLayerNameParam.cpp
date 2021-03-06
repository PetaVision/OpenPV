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
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

InputLayerNameParam::~InputLayerNameParam() {}

void InputLayerNameParam::initialize(char const *name, PVParams *params, Communicator const *comm) {
   LinkedObjectParam::initialize(name, params, comm, std::string("inputLayerName"));
}

void InputLayerNameParam::setObjectType() { mObjectType = "InputLayerNameParam"; }

} // namespace PV
