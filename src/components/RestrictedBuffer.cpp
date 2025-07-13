/*
 * RestrictedBuffer.cpp
 *
 *  Created on: Oct 12, 2018
 *      Author: Pete Schultz
 */

#include "RestrictedBuffer.hpp"

namespace PV {

RestrictedBuffer::RestrictedBuffer(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

RestrictedBuffer::~RestrictedBuffer() {}

void RestrictedBuffer::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   ComponentBuffer::initialize(params, defaults, comm);
   mExtendedFlag = false;
}

void RestrictedBuffer::setObjectType() { mObjectType = "RestrictedBuffer"; }

} // namespace PV
