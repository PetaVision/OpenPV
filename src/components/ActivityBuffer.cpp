/*
 * ActivityBuffer.cpp
 *
 *  Created on: Sep 12, 2018
 *      Author: Pete Schultz
 */

#include "ActivityBuffer.hpp"

namespace PV {

ActivityBuffer::ActivityBuffer(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

ActivityBuffer::~ActivityBuffer() {}

void ActivityBuffer::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   ComponentBuffer::initialize(paramsIO, comm);
   mExtendedFlag = true;
   setBufferLabel("A");
}

void ActivityBuffer::setObjectType() { mObjectType = "ActivityBuffer"; }

} // namespace PV
