/*
 * TestImageActivityComponent.hpp
 *
 *  Created on: Sep 6, 2018
 *      Author: Pete Schultz
 */

#ifndef TESTIMAGEACTIVITYCOMPONENT_HPP_
#define TESTIMAGEACTIVITYCOMPONENT_HPP_

#include "TestImageActivityBuffer.hpp"
#include <components/ActivityComponent.hpp>

namespace PV {

/**
 * The base class for layer buffers such as GSyn, membrane potential, activity, etc.
 */
class TestImageActivityComponent : public ActivityComponent {
  public:
   TestImageActivityComponent(char const *name, HyPerCol *hc);

   virtual ~TestImageActivityComponent();

  protected:
   TestImageActivityComponent() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual void setObjectType();

   virtual ActivityBuffer *createActivity() override;
};

} // namespace PV

#endif // TESTIMAGEACTIVITYCOMPONENT_HPP_
