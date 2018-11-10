/*
 * ActivityBuffer.cpp
 *
 *  Created on: Sep 12, 2018
 *      Author: Pete Schultz
 */

#include "ActivityBuffer.hpp"
#include "columns/HyPerCol.hpp"

namespace PV {

ActivityBuffer::ActivityBuffer(char const *name, HyPerCol *hc) { initialize(name, hc); }

ActivityBuffer::~ActivityBuffer() {}

int ActivityBuffer::initialize(char const *name, HyPerCol *hc) {
   int status    = ComponentBuffer::initialize(name, hc);
   mExtendedFlag = true;
   setBufferLabel("A");
   return status;
}

void ActivityBuffer::setObjectType() { mObjectType = "ActivityBuffer"; }

} // namespace PV
