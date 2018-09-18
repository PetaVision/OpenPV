/*
 * LayerInputBuffer.cpp
 *
 *  Created on: Sep 13, 2018
 *      Author: Pete Schultz
 */

#include "LayerInputBuffer.hpp"
#include "columns/HyPerCol.hpp"

namespace PV {

LayerInputBuffer::LayerInputBuffer(char const *name, HyPerCol *hc) { initialize(name, hc); }

LayerInputBuffer::~LayerInputBuffer() { free(mInitVTypeString); }

int LayerInputBuffer::initialize(char const *name, HyPerCol *hc) {
   int status    = BufferComponent::initialize(name, hc);
   mExtendedFlag = false;
   mBufferLabel  = ""; // GSyn doesn't get checkpointed
   return status;
}

void LayerInputBuffer::setObjectType() { mObjectType = "LayerInputBuffer"; }

int LayerInputBuffer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) { return PV_SUCCESS; }

Response::Status
LayerInputBuffer::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   Response::Status status = BufferComponent::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   // Get GSyn component
   return Response::SUCCESS;
}

void LayerInputBuffer::requireChannel(int channelNeeded) {
   if (channelNeeded >= mNumChannels) {
      mNumChannels = channelNeeded + 1;
   }
}

Response::Status
LayerInputBuffer::registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
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
LayerInputBuffer::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   if (mInitVObject != nullptr) {
      mInitVObject->calcV(mBufferData.data(), getLayerLoc());
   }
   return Response::SUCCESS;
}

void LayerInputBuffer::updateBuffer(double simTime, double deltaTime) {
   // Compute V from GSyn
}

} // namespace PV
