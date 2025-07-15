/*
 * NormalizeNone.cpp
 *
 *  Created on: Oct 24, 2014
 *      Author: pschultz
 */

#include "NormalizeNone.hpp"

namespace PV {

NormalizeNone::NormalizeNone(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   initialize(paramsIO, comm);
}

NormalizeNone::NormalizeNone() {}

NormalizeNone::~NormalizeNone() {}

void NormalizeNone::initialize(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   NormalizeBase::initialize(paramsIO, comm);
}

Response::Status
NormalizeNone::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   return Response::NO_ACTION;
}

} /* namespace PV */
