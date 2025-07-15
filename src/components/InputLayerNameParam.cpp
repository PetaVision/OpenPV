/*
 * InputLayerNameParam.cpp
 *
 *  Created on: Oct 12, 2018
 *      Author: Pete Schultz
 */

#include "InputLayerNameParam.hpp"
#include "observerpattern/ObserverTable.hpp"

namespace PV {

InputLayerNameParam::InputLayerNameParam(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

InputLayerNameParam::~InputLayerNameParam() {}

void InputLayerNameParam::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   LinkedObjectParam::initialize(paramsIO, comm, std::string("inputLayerName"));
}

void InputLayerNameParam::setObjectType() { mObjectType = "InputLayerNameParam"; }

} // namespace PV
