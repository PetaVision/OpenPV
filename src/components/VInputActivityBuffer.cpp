/*
 * VInputActivityBuffer.cpp
 *
 *  Created on: Oct 12, 2018
 *      Author: Pete Schultz
 */

#include "VInputActivityBuffer.hpp"

namespace PV {

VInputActivityBuffer::VInputActivityBuffer(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

VInputActivityBuffer::~VInputActivityBuffer() {}

void VInputActivityBuffer::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   ActivityBuffer::initialize(paramsIO, comm);
}

void VInputActivityBuffer::setObjectType() { mObjectType = "VInputActivityBuffer"; }

Response::Status VInputActivityBuffer::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   Response::Status status = ActivityBuffer::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   mInternalState = message->mObjectTable->findObject<InternalStateBuffer>(getName());
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
