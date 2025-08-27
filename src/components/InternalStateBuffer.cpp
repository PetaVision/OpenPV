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

InternalStateBuffer::InternalStateBuffer(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

InternalStateBuffer::~InternalStateBuffer() {
   delete mInitVObject;
}

void InternalStateBuffer::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   RestrictedBuffer::initialize(paramsIO, comm);
   setBufferLabel("V");
}

void InternalStateBuffer::setObjectType() { mObjectType = "InternalStateBuffer"; }

int InternalStateBuffer::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   ioParam_InitVType(ioSwitch);
   return PV_SUCCESS;
}

void InternalStateBuffer::ioParam_InitVType(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "InitVType", &mInitVTypeString);
   if (ioSwitch == ParamsIOSwitch::Read) {
      BaseObject *object = Factory::instance()->createByKeyword(mInitVTypeString.c_str(), this);
      mInitVObject       = dynamic_cast<BaseInitV *>(object);
      FatalIf(
            mInitVObject == nullptr,
            "%s unable to create InitV object of type %s\n",
            getDescription_c(),
            mInitVObject);
   }
   else if (mInitVObject != nullptr) {
      assert(ioSwitch == ParamsIOSwitch::Write);
      mInitVObject->ioParamsFillGroup(ioSwitch);
      // ioParamsFillGroup(ParamsIOSwitch::Read) is called by mInitVObject constructor in
      // if-clause above. If ioSwitch is ParamsIOSwitch::Read, we don't need to call it again.
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
   return Response::SUCCESS;
}

} // namespace PV
