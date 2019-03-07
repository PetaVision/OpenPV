/*
 * NormalizeNone.cpp
 *
 *  Created on: Oct 24, 2014
 *      Author: pschultz
 */

#include "NormalizeNone.hpp"

namespace PV {

NormalizeNone::NormalizeNone(const char *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

NormalizeNone::NormalizeNone() {}

NormalizeNone::~NormalizeNone() {}

void NormalizeNone::initialize(const char *name, PVParams *params, Communicator const *comm) {
   NormalizeBase::initialize(name, params, comm);
}

Response::Status
NormalizeNone::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   return Response::NO_ACTION;
}

} /* namespace PV */
