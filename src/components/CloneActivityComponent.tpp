/*
 * CloneActivityComponent.cpp
 *
 *  Created on: Oct 12, 2018
 *      Author: Pete Schultz
 *  template implementations for CloneActivityComponent classes.
 *  Note that the .hpp includes this .tpp file at the end;
 *  the .tpp file does not include the .hpp file.
 */

#include "observerpattern/ObserverTable.hpp"

namespace PV {

template <typename V, typename A>
CloneActivityComponent<V, A>::CloneActivityComponent(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

template <typename V, typename A>
CloneActivityComponent<V, A>::~CloneActivityComponent() {}

template <typename V, typename A>
void CloneActivityComponent<V, A>::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   ActivityComponent::initialize(name, params, comm);
}

template <typename V, typename A>
void CloneActivityComponent<V, A>::setObjectType() {
   mObjectType = "CloneActivityComponent";
}

template <typename V, typename A>
void CloneActivityComponent<V, A>::fillComponentTable() {
   ActivityComponent::fillComponentTable(); // creates Activity
   mInternalState = createInternalState();
   if (mInternalState) {
      addUniqueComponent(mInternalState);
   }
}

template <typename V, typename A>
ActivityBuffer *CloneActivityComponent<V, A>::createActivity() {
   return new A(getName(), parameters(), mCommunicator);
}

template <typename V, typename A>
InternalStateBuffer *CloneActivityComponent<V, A>::createInternalState() {
   return new V(getName(), parameters(), mCommunicator);
}

template <typename V, typename A>
Response::Status CloneActivityComponent<V, A>::initializeState(
      std::shared_ptr<InitializeStateMessage const> message) {
   if (mInternalState) {
      mInternalState->respond(message);
   }
   mActivity->respond(message);
   return Response::SUCCESS;
}

template <typename V, typename A>
Response::Status CloneActivityComponent<V, A>::updateActivity(double simTime, double deltaTime) {
   if (mInternalState) {
      mInternalState->updateBuffer(simTime, deltaTime);
   }
   mActivity->updateBuffer(simTime, deltaTime);
   return Response::SUCCESS;
}

} // namespace PV
