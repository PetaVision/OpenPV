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
#include "columns/HyPerCol.hpp"

namespace PV {

template <typename A>
ActivityComponentActivityOnly<A>::ActivityComponentActivityOnly(char const *name, HyPerCol *hc) {
   initialize(name, hc);
}

template <typename A>
ActivityComponentActivityOnly<A>::~ActivityComponentActivityOnly() {}

template <typename A>
int ActivityComponentActivityOnly<A>::initialize(char const *name, HyPerCol *hc) {
   return ActivityComponent::initialize(name, hc);
}

template <typename A>
void ActivityComponentActivityOnly<A>::setObjectType() {
   mObjectType = "ActivityComponentActivityOnly";
}

template <typename A>
ActivityBuffer *ActivityComponentActivityOnly<A>::createActivity() {
   return new A(getName(), parent);
}

} // namespace PV
