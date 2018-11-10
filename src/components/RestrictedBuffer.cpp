/*
 * RestrictedBuffer.cpp
 *
 *  Created on: Oct 12, 2018
 *      Author: Pete Schultz
 */

#include "RestrictedBuffer.hpp"
#include "columns/HyPerCol.hpp"

namespace PV {

RestrictedBuffer::RestrictedBuffer(char const *name, HyPerCol *hc) { initialize(name, hc); }

RestrictedBuffer::~RestrictedBuffer() {}

int RestrictedBuffer::initialize(char const *name, HyPerCol *hc) {
   int status    = ComponentBuffer::initialize(name, hc);
   mExtendedFlag = false;
   return status;
}

void RestrictedBuffer::setObjectType() { mObjectType = "RestrictedBuffer"; }

} // namespace PV
