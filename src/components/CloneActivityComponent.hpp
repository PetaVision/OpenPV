/*
 * CloneActivityComponent.hpp
 *
 *  Created on: Oct 12, 2018
 *      Author: Pete Schultz
 */

#ifndef CLONEACTIVITYCOMPONENT_HPP_
#define CLONEACTIVITYCOMPONENT_HPP_

#include "components/ActivityComponent.hpp"
#include "components/GSynAccumulator.hpp"
#include "components/InternalStateBuffer.hpp"

namespace PV {

/**
 * The class template for ActivityComponent classes that use a V component and
 * an A component (derived from InternalStateBuffer and ActivityBuffer, respectively)
 * but do not use a GSynAccumulator-derived component. This is useful for
 * cases where V does not get its input from GSyn, for example CloneInternalStateBuffer.
 */
template <typename V, typename A>
class CloneActivityComponent : public ActivityComponent {
  public:
   CloneActivityComponent(char const *name, PVParams *params, Communicator const *comm);

   virtual ~CloneActivityComponent();

  protected:
   CloneActivityComponent() {}

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual void setObjectType() override;

   virtual void fillComponentTable() override;

   virtual ActivityBuffer *createActivity() override;

   virtual InternalStateBuffer *createInternalState();

   /**
    * Calls the initializeState methods of AccumulatedGSyn, InternalState, and Activity,
    * in that order.
    */
   virtual Response::Status
   initializeState(std::shared_ptr<InitializeStateMessage const> message) override;

   /**
    * Calls the updateBuffer methods of AccumulatedGSyn, InternalState, and Activity,
    * in that order.
    */
   virtual Response::Status updateActivity(double simTime, double deltaTime) override;

  protected:
   InternalStateBuffer *mInternalState = nullptr;
};

} // namespace PV

#include "CloneActivityComponent.tpp"

#endif // CLONEACTIVITYCOMPONENT_HPP_
