/*
 * ActivityBuffer.cpp
 *
 *  Created on: Sep 6, 2018
 *      Author: Pete Schultz
 */

#include "ActivityBuffer.hpp"
#include "columns/HyPerCol.hpp"

#ifdef PV_USE_CUDA
#undef PV_USE_CUDA
#include <layers/updateStateFunctions.h>
#define PV_USE_CUDA
#else
#include <layers/updateStateFunctions.h>
#endif // PV_USE_CUDA

namespace PV {

ActivityBuffer::ActivityBuffer(char const *name, HyPerCol *hc) { initialize(name, hc); }

ActivityBuffer::~ActivityBuffer() {}

int ActivityBuffer::initialize(char const *name, HyPerCol *hc) {
   int status    = BufferComponent::initialize(name, hc);
   mExtendedFlag = true;
   mBufferLabel  = "A";
   return status;
}

void ActivityBuffer::setObjectType() { mObjectType = "ActivityBuffer"; }

Response::Status
ActivityBuffer::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   Response::Status status = BufferComponent::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   //    mInternalState = mapLookupByType<InternalStateBuffer>(message->mHierarchy);
   //    FatalIf(!mInternalState, "%s requires an InternalStateBuffer component.\n",
   //    getDescription_c());
   //    if (!mInternalState->getInitInfoCommunicatedFlag()) {
   //       return Response::POSTPONE;
   //    }
   //    PVLayerLoc const *internalStateLoc = mInternalState->getLayerLoc();
   //    PVLayerLoc const *activityLoc      = getLayerLoc();
   //    checkDimensions(internalStateLoc->nx, activityLoc->nx, "nx");
   //    checkDimensions(internalStateLoc->ny, activityLoc->ny, "ny");
   //    checkDimensions(internalStateLoc->nf, activityLoc->nf, "nf");
   //    checkDimensions(internalStateLoc->nbatch, activityLoc->nbatch, "nbatch");
   return Response::SUCCESS;
}

// void ActivityBuffer::checkDimensions(
//       int internalStateSize,
//       int activitySize,
//       char const *fieldname) {
//    FatalIf(
//          internalStateSize != activitySize,
//          "%s and %s do not have the same %s (%d versus %d)\n",
//          mInternalState->getDescription(),
//          getDescription(),
//          fieldname,
//          internalStateSize,
//          activitySize);
// }

Response::Status
ActivityBuffer::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   Response::Status status = BufferComponent::initializeState(message);
   if (!Response::completed(status)) {
      return status;
   }
   //    if (!mInternalState->getInitialValuesSetFlag()) {
   //       return Response::POSTPONE;
   //    }
   //    setActivity();
   return Response::SUCCESS;
}

void ActivityBuffer::updateState(double simTime, double deltaTime) {
   //    mInternalState->updateState(simTime, deltaTime);
   //    setActivity();
}

void ActivityBuffer::setActivity() {
   //    PVLayerLoc const *loc = getLayerLoc();
   //    float const *V        = mInternalState->getBufferData();
   //    setActivity_HyPerLayer(
   //          loc->nbatch,
   //          getBufferSize(),
   //          mBufferData.data(),
   //          mInternalState->getBufferData(),
   //          loc->nx,
   //          loc->ny,
   //          loc->nf,
   //          loc->halo.lt,
   //          loc->halo.rt,
   //          loc->halo.dn,
   //          loc->halo.up);
}

} // namespace PV
