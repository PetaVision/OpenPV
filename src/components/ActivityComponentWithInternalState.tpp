/*
 * ActivityComponentWithInternalState.cpp
 *
 *  Created on: Oct 12, 2018
 *      Author: Pete Schultz
 *  template implementations for ActivityComponentWithInternalState classes.
 *  Note that the .hpp includes this .tpp file at the end;
 *  the .tpp file does not include the .hpp file.
 */

#include "ActivityComponentWithInternalState.hpp"
#include "columns/HyPerCol.hpp"

namespace PV {

template <typename V, typename A>
ActivityComponentWithInternalState<V, A>::ActivityComponentWithInternalState(
      char const *name,
      PVParams *params,
      Communicator *comm) {
   initialize(name, params, comm);
}

template <typename V, typename A>
ActivityComponentWithInternalState<V, A>::~ActivityComponentWithInternalState() {}

template <typename V, typename A>
void ActivityComponentWithInternalState<V, A>::initialize(
      char const *name,
      PVParams *params,
      Communicator *comm) {
   HyPerActivityComponent::initialize(name, params, comm);
}

template <typename V, typename A>
void ActivityComponentWithInternalState<V, A>::setObjectType() {
   mObjectType = "ActivityComponentWithInternalState";
}

template <typename V, typename A>
InternalStateBuffer *ActivityComponentWithInternalState<V, A>::createInternalState() {
   return new V(getName(), parameters(), mCommunicator);
}

template <typename V, typename A>
ActivityBuffer *ActivityComponentWithInternalState<V, A>::createActivity() {
   return new A(getName(), parameters(), mCommunicator);
}

} // namespace PV
