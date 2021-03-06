/*
 * HyPerActivityComponent.hpp
 *
 *  Created on: Oct 12, 2018
 *      Author: Pete Schultz
 */

#ifndef HYPERACTIVITYCOMPONENT_HPP_
#define HYPERACTIVITYCOMPONENT_HPP_

#include "components/ActivityComponent.hpp"
#include "components/GSynAccumulator.hpp"
#include "components/InternalStateBuffer.hpp"

namespace PV {

/**
 * The class template for ActivityComponent classes that use an accumulated-GSyn component,
 * V component, and A component (derived from GSynAccumulator, InternalStateBuffer,
 * and ActivityBuffer, respectively).
 */
template <typename G, typename V, typename A>
class HyPerActivityComponent : public ActivityComponent {
  public:
   HyPerActivityComponent(char const *name, PVParams *params, Communicator const *comm);

   virtual ~HyPerActivityComponent();

  protected:
   HyPerActivityComponent() {}

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual void setObjectType() override;

   virtual void fillComponentTable() override;

   virtual ActivityBuffer *createActivity() override;

   virtual InternalStateBuffer *createInternalState();

   virtual GSynAccumulator *createAccumulatedGSyn();

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
   GSynAccumulator *mAccumulatedGSyn   = nullptr;
};

} // namespace PV

#include "HyPerActivityComponent.tpp"

#endif // HYPERACTIVITYCOMPONENT_HPP_
