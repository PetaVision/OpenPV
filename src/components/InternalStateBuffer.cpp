/*
 * InternalStateBuffer.cpp
 *
 *  Created on: Sep 11, 2018
 *      Author: Pete Schultz
 */

#include "InternalStateBuffer.hpp"
#include "columns/Factory.hpp"
#include <cassert>

namespace PV {

InternalStateBuffer::InternalStateBuffer(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

InternalStateBuffer::~InternalStateBuffer() {
   free(mInitVTypeString);
   delete mInitVObject;
}

void InternalStateBuffer::initialize(char const *name, PVParams *params, Communicator const *comm) {
   RestrictedBuffer::initialize(name, params, comm);
   setBufferLabel("V");
}

void InternalStateBuffer::setObjectType() { mObjectType = "InternalStateBuffer"; }

int InternalStateBuffer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_InitVType(ioFlag);
   return PV_SUCCESS;
}

void InternalStateBuffer::ioParam_InitVType(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamString(
         ioFlag,
         getName(),
         "InitVType",
         &mInitVTypeString,
         BaseInitV::mDefaultInitV.data(),
         true /*warnIfAbsent*/);
   if (ioFlag == PARAMS_IO_READ) {
      BaseObject *object = Factory::instance()->createByKeyword(mInitVTypeString, this);
      mInitVObject       = dynamic_cast<BaseInitV *>(object);
      FatalIf(
            mInitVObject == nullptr,
            "%s unable to create InitV object of type %s\n",
            getDescription_c(),
            mInitVObject);
   }
   else if (mInitVObject != nullptr) {
      assert(ioFlag == PARAMS_IO_WRITE);
      mInitVObject->ioParamsFillGroup(ioFlag);
      // ioParamsFillGroup(PARAMS_IO_READ) is called by mInitVObject constructor in if-clause above.
      // If ioFlag is PARAMS_IO_READ, we don't need to call it again.
   }
}

Response::Status InternalStateBuffer::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   Response::Status status = RestrictedBuffer::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   if (mInitVObject) {
      status = mInitVObject->respond(message);
      if (!Response::completed(status)) {
         return status;
      }
   }

   return Response::SUCCESS;
}

Response::Status InternalStateBuffer::registerData(
      std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   auto status = RestrictedBuffer::registerData(message);
   if (!Response::completed(status)) {
      return status;
   }
   if (mInitVObject != nullptr) {
      mInitVObject->respond(message);
   }
   return Response::SUCCESS;
}

Response::Status
InternalStateBuffer::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   if (mInitVObject != nullptr) {
      mInitVObject->calcV(mBufferData.data(), getLayerLoc());
   }
   if (getLayerLoc()->bcast) {
      MPI_Bcast(
            mBufferData.data(),
            int(mBufferData.size()),
            MPI_FLOAT,
            0 /*root*/,
            getCommunicator()->communicator());
   }
   return Response::SUCCESS;
}

} // namespace PV
