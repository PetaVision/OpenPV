/*
 * InternalStateBuffer.cpp
 *
 *  Created on: Sep 11, 2018
 *      Author: Pete Schultz
 */

#include "InternalStateBuffer.hpp"
#include "columns/Factory.hpp"

namespace PV {

InternalStateBuffer::InternalStateBuffer(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

InternalStateBuffer::~InternalStateBuffer() {
   delete mInitVObject;
}

void InternalStateBuffer::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   RestrictedBuffer::initialize(params, defaults, comm);
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
   if (mInitVObject != nullptr) {
      if (ioSwitch == ParamsIOSwitch::Write) {
         mInitVObject->getParamsIO()->setPrintParamsStream(mParamsIO->getPrintParamsStream());
         mInitVObject->getParamsIO()->setPrintLuaStream(mParamsIO->getPrintLuaStream());
      }
      mInitVObject->ioParamsFillGroup(ioSwitch);
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
