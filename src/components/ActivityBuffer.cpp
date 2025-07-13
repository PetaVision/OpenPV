/*
 * ActivityBuffer.cpp
 *
 *  Created on: Sep 12, 2018
 *      Author: Pete Schultz
 */

#include "ActivityBuffer.hpp"

namespace PV {

ActivityBuffer::ActivityBuffer(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

ActivityBuffer::~ActivityBuffer() {}

void ActivityBuffer::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   ComponentBuffer::initialize(params, defaults, comm);
   mExtendedFlag = true;
   setBufferLabel("A");
}

void ActivityBuffer::setObjectType() { mObjectType = "ActivityBuffer"; }

} // namespace PV
