/*
 * InternalStateBuffer.cpp
 *
 *  Created on: Sep 6, 2018
 *      Author: Pete Schultz
 */

#include "InternalStateBuffer.hpp"
#include "columns/HyPerCol.hpp"

namespace PV {

InternalStateBuffer::InternalStateBuffer(char const *name, HyPerCol *hc) { initialize(name, hc); }

InternalStateBuffer::~InternalStateBuffer() { free(mInitVTypeString); }

int InternalStateBuffer::initialize(char const *name, HyPerCol *hc) {
   int status    = BufferComponent::initialize(name, hc);
   mExtendedFlag = false;
   mBufferLabel  = "V";
   return status;
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
      BaseObject *object = Factory::instance()->createByKeyword(mInitVTypeString, name, parent);
      mInitVObject       = dynamic_cast<BaseInitV *>(object);
      if (mInitVObject == nullptr) {
         ErrorLog().printf("%s: unable to create InitV object\n", getDescription_c());
         abort();
      }
   }
   if (mInitVObject != nullptr) {
      mInitVObject->ioParamsFillGroup(ioFlag);
   }
}

Response::Status InternalStateBuffer::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   Response::Status status = BufferComponent::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   // Get GSyn component
   return Response::SUCCESS;
}

// void checkDimensions(int gSynSize, int internalStateSize, char const *fieldname) {
//    FatalIf(
//          gSynSize != internalStateSize,
//          "%s and %s do not have the same %s (%d versus %d)\n",
//          mGSyn->getDescription(),
//          getDescription(),
//          fieldname,
//          internalStateSize,
//          activitySize);
// }

Response::Status InternalStateBuffer::registerData(
      std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   auto status = BufferComponent::registerData(message);
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

void InternalStateBuffer::updateState(double simTime, double deltaTime) {
   // Compute V from GSyn
}

} // namespace PV
