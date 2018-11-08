/*
 * ActivityComponentActivityOnly.hpp
 *
 *  Created on: Oct 12, 2018
 *      Author: Pete Schultz
 */

#ifndef ACTIVITYCOMPONENTACTIVITYONLY_HPP_
#define ACTIVITYCOMPONENTACTIVITYONLY_HPP_

#include "ActivityComponent.hpp"

namespace PV {

/**
 * The class template for ActivityComponent classes that need an ActivityBuffer component but
 * not an InternalStateBuffer component.
 */
template <typename A>
class ActivityComponentActivityOnly : public ActivityComponent {
  public:
   ActivityComponentActivityOnly(char const *name, PVParams *params, Communicator *comm);

   virtual ~ActivityComponentActivityOnly();

  protected:
   ActivityComponentActivityOnly() {}

   void initialize(char const *name, PVParams *params, Communicator *comm);

   virtual void setObjectType();

   virtual ActivityBuffer *createActivity() override;
};

} // namespace PV

#include "ActivityComponentActivityOnly.tpp"

#endif // ACTIVITYCOMPONENTACTIVITYONLY_HPP_
