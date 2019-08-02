/*
 * HyPerActivityComponent.cpp
 *
 *  Created on: Oct 12, 2018
 *      Author: Pete Schultz
 *  template implementations for HyPerActivityComponent classes.
 *  Note that the .hpp includes this .tpp file at the end;
 *  the .tpp file does not include the .hpp file.
 */

#include "observerpattern/ObserverTable.hpp"

namespace PV {

template <typename G, typename V, typename A>
HyPerActivityComponent<G, V, A>::HyPerActivityComponent(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

template <typename G, typename V, typename A>
HyPerActivityComponent<G, V, A>::~HyPerActivityComponent() {}

template <typename G, typename V, typename A>
void HyPerActivityComponent<G, V, A>::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   ActivityComponent::initialize(name, params, comm);
}

template <typename G, typename V, typename A>
void HyPerActivityComponent<G, V, A>::setObjectType() {
   mObjectType = "HyPerActivityComponent";
}

template <typename G, typename V, typename A>
void HyPerActivityComponent<G, V, A>::fillComponentTable() {
   ActivityComponent::fillComponentTable(); // creates Activity
   mInternalState = createInternalState();
   if (mInternalState) {
      addUniqueComponent(mInternalState);
   }
   mAccumulatedGSyn = createAccumulatedGSyn();
   if (mAccumulatedGSyn) {
      addUniqueComponent(mAccumulatedGSyn);
   }
}

template <typename G, typename V, typename A>
ActivityBuffer *HyPerActivityComponent<G, V, A>::createActivity() {
   return new A(getName(), parameters(), mCommunicator);
}

template <typename G, typename V, typename A>
InternalStateBuffer *HyPerActivityComponent<G, V, A>::createInternalState() {
   return new V(getName(), parameters(), mCommunicator);
}

template <typename G, typename V, typename A>
GSynAccumulator *HyPerActivityComponent<G, V, A>::createAccumulatedGSyn() {
   return new G(getName(), parameters(), mCommunicator);
}

template <typename G, typename V, typename A>
Response::Status HyPerActivityComponent<G, V, A>::initializeState(
      std::shared_ptr<InitializeStateMessage const> message) {
   if (mAccumulatedGSyn) {
      mAccumulatedGSyn->respond(message);
   }
   if (mInternalState) {
      mInternalState->respond(message);
   }
   mActivity->respond(message);
   return Response::SUCCESS;
}

template <typename G, typename V, typename A>
Response::Status HyPerActivityComponent<G, V, A>::updateActivity(double simTime, double deltaTime) {
   if (mAccumulatedGSyn) {
      mAccumulatedGSyn->updateBuffer(simTime, deltaTime);
   }
   if (mInternalState) {
      mInternalState->updateBuffer(simTime, deltaTime);
   }
   mActivity->updateBuffer(simTime, deltaTime);
   return Response::SUCCESS;
}

} // namespace PV
