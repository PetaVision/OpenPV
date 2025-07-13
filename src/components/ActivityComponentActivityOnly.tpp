/*
 * ActivityComponentActivityOnly.cpp
 *
 *  Created on: Oct 12, 2018
 *      Author: Pete Schultz
 *  template implementations for ActivityComponentActivityOnly classes.
 *  Note that the .hpp includes this .tpp file at the end;
 *  the .tpp file does not include the .hpp file.
 */

namespace PV {

template <typename A>
ActivityComponentActivityOnly<A>::ActivityComponentActivityOnly(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

template <typename A>
ActivityComponentActivityOnly<A>::~ActivityComponentActivityOnly() {}

template <typename A>
void ActivityComponentActivityOnly<A>::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   ActivityComponent::initialize(params, defaults, comm);
}

template <typename A>
void ActivityComponentActivityOnly<A>::setObjectType() {
   mObjectType = "ActivityComponentActivityOnly";
}

template <typename A>
ActivityBuffer *ActivityComponentActivityOnly<A>::createActivity() {
   return new A(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

} // namespace PV
