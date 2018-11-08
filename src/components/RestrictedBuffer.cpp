/*
 * RestrictedBuffer.cpp
 *
 *  Created on: Oct 12, 2018
 *      Author: Pete Schultz
 */

#include "RestrictedBuffer.hpp"
#include "columns/HyPerCol.hpp"

namespace PV {

RestrictedBuffer::RestrictedBuffer(char const *name, PVParams *params, Communicator *comm) {
   initialize(name, params, comm);
}

RestrictedBuffer::~RestrictedBuffer() {}

void RestrictedBuffer::initialize(char const *name, PVParams *params, Communicator *comm) {
   ComponentBuffer::initialize(name, params, comm);
   mExtendedFlag = false;
}

void RestrictedBuffer::setObjectType() { mObjectType = "RestrictedBuffer"; }

} // namespace PV
