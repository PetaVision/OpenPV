/*
 * HyPerActivityComponent.hpp
 *
 *  Created on: Oct 12, 2018
 *      Author: Pete Schultz
 */

#ifndef HYPERACTIVITYCOMPONENT_HPP_
#define HYPERACTIVITYCOMPONENT_HPP_

#include "components/ActivityComponent.hpp"
#include "components/InternalStateBuffer.hpp"

namespace PV {

/**
 * The base class for layer buffers such as GSyn, membrane potential, activity, etc.
 */
class HyPerActivityComponent : public ActivityComponent {
  public:
   HyPerActivityComponent(char const *name, PVParams *params, Communicator *comm);

   virtual ~HyPerActivityComponent();

   virtual Response::Status updateActivity(double simTime, double deltaTime) override;

  protected:
   HyPerActivityComponent() {}

   void initialize(char const *name, PVParams *params, Communicator *comm);

   virtual void setObjectType();

   virtual void createComponentTable(char const *tableDescription) override;

   virtual ActivityBuffer *createActivity() override;

   virtual InternalStateBuffer *createInternalState();

   virtual Response::Status
   initializeState(std::shared_ptr<InitializeStateMessage const> message) override;

  protected:
   InternalStateBuffer *mInternalState = nullptr;
};

} // namespace PV

#endif // HYPERACTIVITYCOMPONENT_HPP_
