/*
 * ActivityBuffer.cpp
 *
 *  Created on: Sep 12, 2018
 *      Author: Pete Schultz
 */

#include "ActivityBuffer.hpp"

namespace PV {

ActivityBuffer::ActivityBuffer(char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

ActivityBuffer::~ActivityBuffer() {}

void ActivityBuffer::initialize(char const *name, PVParams *params, Communicator const *comm) {
   ComponentBuffer::initialize(name, params, comm);
   mExtendedFlag = true;
   setBufferLabel("A");
}

void ActivityBuffer::setObjectType() { mObjectType = "ActivityBuffer"; }

} // namespace PV
