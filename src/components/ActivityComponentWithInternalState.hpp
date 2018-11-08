/*
 * ActivityComponentWithInternalState.hpp
 *
 *  Created on: Oct 12, 2018
 *      Author: Pete Schultz
 */

#ifndef ACTIVITYCOMPONENTWITHINTERNALSTATE_HPP_
#define ACTIVITYCOMPONENTWITHINTERNALSTATE_HPP_

#include "HyPerActivityComponent.hpp"

namespace PV {

/**
 * The class template for ActivityComponent classes that need an InternalStateBuffer component
 * as well as an ActivityBuffer component.
 */
template <typename V, typename A>
class ActivityComponentWithInternalState : public HyPerActivityComponent {
  public:
   ActivityComponentWithInternalState(char const *name, PVParams *params, Communicator *comm);

   virtual ~ActivityComponentWithInternalState();

  protected:
   ActivityComponentWithInternalState() {}

   void initialize(char const *name, PVParams *params, Communicator *comm);

   virtual void setObjectType();

   virtual InternalStateBuffer *createInternalState() override;

   virtual ActivityBuffer *createActivity() override;
};

} // namespace PV

#include "ActivityComponentWithInternalState.tpp"

#endif // ACTIVITYCOMPONENTWITHINTERNALSTATE_HPP_
