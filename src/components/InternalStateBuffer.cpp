/*
 * InternalStateBuffer.cpp
 *
 *  Created on: Sep 11, 2018
 *      Author: Pete Schultz
 */

#include "InternalStateBuffer.hpp"
#include "columns/HyPerCol.hpp"

namespace PV {

InternalStateBuffer::InternalStateBuffer(char const *name, PVParams *params, Communicator *comm) {
   initialize(name, params, comm);
}

InternalStateBuffer::~InternalStateBuffer() { free(mInitVTypeString); }

void InternalStateBuffer::initialize(char const *name, PVParams *params, Communicator *comm) {
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
         name,
         "InitVType",
         &mInitVTypeString,
         BaseInitV::mDefaultInitV.data(),
         true /*warnIfAbsent*/);
   if (ioFlag == PARAMS_IO_READ) {
      BaseObject *object = Factory::instance()->createByKeyword(
            mInitVTypeString, name, parameters(), mCommunicator);
      mInitVObject = dynamic_cast<BaseInitV *>(object);
      if (mInitVObject == nullptr) {
         ErrorLog().printf("%s: unable to create InitV object\n", getDescription_c());
         abort();
      }
   }
   if (mInitVObject != nullptr) {
      mInitVObject->ioParamsFillGroup(ioFlag);
   }
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
   return Response::SUCCESS;
}

} // namespace PV
