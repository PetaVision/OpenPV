/*
 * VInputActivityBuffer.cpp
 *
 *  Created on: Oct 12, 2018
 *      Author: Pete Schultz
 */

#include "VInputActivityBuffer.hpp"
#include "columns/HyPerCol.hpp"

namespace PV {

VInputActivityBuffer::VInputActivityBuffer(char const *name, HyPerCol *hc) { initialize(name, hc); }

VInputActivityBuffer::~VInputActivityBuffer() {}

int VInputActivityBuffer::initialize(char const *name, HyPerCol *hc) {
   int status = ActivityBuffer::initialize(name, hc);
   return status;
}

void VInputActivityBuffer::setObjectType() { mObjectType = "VInputActivityBuffer"; }

Response::Status VInputActivityBuffer::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   Response::Status status = ActivityBuffer::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   mInternalState = message->mHierarchy->lookupByType<InternalStateBuffer>();
   FatalIf(
         mInternalState == nullptr,
         "%s could not find an InternalStateBuffer component.\n",
         getDescription_c());
   if (!mInternalState->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }
   checkDimensionsEqual(mInternalState, this);
   return Response::SUCCESS;
}

} // namespace PV
