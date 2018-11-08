/*
 * BackgroundActivityComponent.hpp
 *
 *  Created on: 4/16/15
 *  slundquist
 */

#ifndef BACKGROUNDACTIVITYCOMPONENT_HPP_
#define BACKGROUNDACTIVITYCOMPONENT_HPP_

#include "components/ActivityComponent.hpp"

namespace PV {

/**
 * The base class for layer buffers such as GSyn, membrane potential, activity, etc.
 */
class BackgroundActivityComponent : public ActivityComponent {

  public:
   BackgroundActivityComponent(char const *name, PVParams *params, Communicator *comm);

   virtual ~BackgroundActivityComponent();

  protected:
   BackgroundActivityComponent() {}

   void initialize(char const *name, PVParams *params, Communicator *comm);

   virtual void setObjectType();

   virtual InternalStateBuffer *createInternalStateBuffer();

   virtual InternalStateUpdater *createInternalStateUpdater();

   virtual ActivityUpdater *createActivityUpdater();

  protected:
   // Data members would go here
};

} // namespace PV

#endif // BACKGROUNDACTIVITYCOMPONENT_HPP_
