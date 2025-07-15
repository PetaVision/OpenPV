/*
 * RestrictedBuffer.cpp
 *
 *  Created on: Oct 12, 2018
 *      Author: Pete Schultz
 */

#include "RestrictedBuffer.hpp"

namespace PV {

RestrictedBuffer::RestrictedBuffer(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

RestrictedBuffer::~RestrictedBuffer() {}

void RestrictedBuffer::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   ComponentBuffer::initialize(paramsIO, comm);
   mExtendedFlag = false;
}

void RestrictedBuffer::setObjectType() { mObjectType = "RestrictedBuffer"; }

} // namespace PV
