/*
 * NormalizeNone.cpp
 *
 *  Created on: Oct 24, 2014
 *      Author: pschultz
 */

#include "NormalizeNone.hpp"

namespace PV {

NormalizeNone::NormalizeNone(const char *name, HyPerCol *hc) { initialize(name, hc); }

NormalizeNone::NormalizeNone() {}

NormalizeNone::~NormalizeNone() {}

int NormalizeNone::initialize(const char *name, HyPerCol *hc) {
   return NormalizeBase::initialize(name, hc);
}

Response::Status
NormalizeNone::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   return Response::NO_ACTION;
}

} /* namespace PV */
