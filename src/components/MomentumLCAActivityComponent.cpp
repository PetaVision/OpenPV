/*
 * MomentumLCAActivityComponent.cpp
 *
 *  Created on: Mar 15, 2016
 *      Author: slundquist
 */

#include "MomentumLCAActivityComponent.hpp"
#include "checkpointing/CheckpointEntryRandState.hpp"
#include "columns/HyPerCol.hpp"
#include "components/ANNActivityBuffer.hpp"
#include "components/MomentumLCAInternalStateBuffer.hpp"
#include "observerpattern/ObserverTable.hpp"

namespace PV {

MomentumLCAActivityComponent::MomentumLCAActivityComponent(char const *name, HyPerCol *hc) {
   initialize(name, hc);
}

MomentumLCAActivityComponent::~MomentumLCAActivityComponent() {}

int MomentumLCAActivityComponent::initialize(char const *name, HyPerCol *hc) {
   return HyPerActivityComponent::initialize(name, hc);
}

void MomentumLCAActivityComponent::setObjectType() { mObjectType = "MomentumLCAActivityComponent"; }

void MomentumLCAActivityComponent::createComponentTable(char const *tableDescription) {
   HyPerActivityComponent::createComponentTable(tableDescription); // creates A and V buffers
   mPrevDrive = createPrevDrive();
   if (mPrevDrive) {
      addObserver(mPrevDrive->getDescription(), mPrevDrive);
   }
}

ActivityBuffer *MomentumLCAActivityComponent::createActivity() {
   return new ANNActivityBuffer(getName(), parent);
}

InternalStateBuffer *MomentumLCAActivityComponent::createInternalState() {
   return new MomentumLCAInternalStateBuffer(getName(), parent);
}

RestrictedBuffer *MomentumLCAActivityComponent::createPrevDrive() {
   RestrictedBuffer *buffer = new RestrictedBuffer(getName(), parent);
   buffer->setBufferLabel("prevDrive");
   return buffer;
}

Response::Status MomentumLCAActivityComponent::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   return HyPerActivityComponent::communicateInitInfo(message);
}

Response::Status MomentumLCAActivityComponent::allocateDataStructures() {
   auto status = HyPerActivityComponent::allocateDataStructures();
   if (!Response::completed(status)) {
      return status;
   }
   ComponentBuffer::checkDimensionsEqual(mInternalState, mPrevDrive);
   return Response::SUCCESS;
}
Response::Status MomentumLCAActivityComponent::registerData(
      std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   return HyPerActivityComponent::registerData(message);
}

Response::Status MomentumLCAActivityComponent::initializeState(
      std::shared_ptr<InitializeStateMessage const> message) {
   auto status = HyPerActivityComponent::initializeState(message);
   if (!Response::completed(status)) {
      return status;
   }
   float *prevDrive                = mPrevDrive->getReadWritePointer();
   int const numNeuronsAcrossBatch = mPrevDrive->getBufferSizeAcrossBatch();
   for (int k = 0; k < numNeuronsAcrossBatch; k++) {
      prevDrive[k] = 0.0f;
   }
   return Response::SUCCESS;
}

} // namespace PV
