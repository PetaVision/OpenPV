/*
 * MomentumLCAActivityComponent.cpp
 *
 *  Created on: Mar 15, 2016
 *      Author: slundquist
 */

#include "MomentumLCAActivityComponent.hpp"
#include "checkpointing/CheckpointEntryRandState.hpp"
#include "components/ANNActivityBuffer.hpp"
#include "components/MomentumLCAInternalStateBuffer.hpp"
#include "observerpattern/ObserverTable.hpp"

namespace PV {

MomentumLCAActivityComponent::MomentumLCAActivityComponent(
      char const *name,
      PVParams *params,
      Communicator *comm) {
   initialize(name, params, comm);
}

MomentumLCAActivityComponent::~MomentumLCAActivityComponent() {}

void MomentumLCAActivityComponent::initialize(
      char const *name,
      PVParams *params,
      Communicator *comm) {
   BaseMomentumActivityComponent::initialize(name, params, comm);
}

void MomentumLCAActivityComponent::setObjectType() { mObjectType = "MomentumLCAActivityComponent"; }

void MomentumLCAActivityComponent::createComponentTable(char const *tableDescription) {
   BaseMomentumActivityComponent::createComponentTable(tableDescription);
   // creates A, V, and accumulated-GSyn buffers.
   mPrevDrive = createPrevDrive();
   if (mPrevDrive) {
      addObserver(mPrevDrive->getDescription(), mPrevDrive);
   }
}

RestrictedBuffer *MomentumLCAActivityComponent::createPrevDrive() {
   RestrictedBuffer *buffer = new RestrictedBuffer(getName(), parameters(), mCommunicator);
   buffer->setBufferLabel("prevDrive");
   return buffer;
}

Response::Status MomentumLCAActivityComponent::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   return BaseMomentumActivityComponent::communicateInitInfo(message);
}

Response::Status MomentumLCAActivityComponent::allocateDataStructures() {
   auto status = BaseMomentumActivityComponent::allocateDataStructures();
   if (!Response::completed(status)) {
      return status;
   }
   ComponentBuffer::checkDimensionsEqual(mInternalState, mPrevDrive);
   return Response::SUCCESS;
}
Response::Status MomentumLCAActivityComponent::registerData(
      std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   return BaseMomentumActivityComponent::registerData(message);
}

Response::Status MomentumLCAActivityComponent::initializeState(
      std::shared_ptr<InitializeStateMessage const> message) {
   auto status = BaseMomentumActivityComponent::initializeState(message);
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
