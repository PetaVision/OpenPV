/*
 * ActivityComponentActivityOnly.cpp
 *
 *  Created on: Oct 12, 2018
 *      Author: Pete Schultz
 *  template implementations for ActivityComponentActivityOnly classes.
 *  Note that the .hpp includes this .tpp file at the end;
 *  the .tpp file does not include the .hpp file.
 */

#include "ActivityComponentActivityOnly.hpp"

namespace PV {

template <typename A>
ActivityComponentActivityOnly<A>::ActivityComponentActivityOnly(
      char const *name,
      PVParams *params,
      Communicator *comm) {
   initialize(name, params, comm);
}

template <typename A>
ActivityComponentActivityOnly<A>::~ActivityComponentActivityOnly() {}

template <typename A>
void ActivityComponentActivityOnly<A>::initialize(
      char const *name,
      PVParams *params,
      Communicator *comm) {
   ActivityComponent::initialize(name, params, comm);
}

template <typename A>
void ActivityComponentActivityOnly<A>::setObjectType() {
   mObjectType = "ActivityComponentActivityOnly";
}

template <typename A>
ActivityBuffer *ActivityComponentActivityOnly<A>::createActivity() {
   return new A(getName(), parameters(), mCommunicator);
}

} // namespace PV
