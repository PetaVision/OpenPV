/*
 * MomentumLCAActivityComponent.hpp
 *
 *  Created on: Mar 15, 2016
 *      Author: slundquist
 */

#ifndef MOMENTUMLCAACTIVITYCOMPONENT_HPP_
#define MOMENTUMLCAACTIVITYCOMPONENT_HPP_

#include "components/HyPerActivityComponent.hpp"

#include "columns/Random.hpp"
#include "components/ActivityBuffer.hpp"
#include "components/InternalStateBuffer.hpp"
#include "components/LIFLayerInputBuffer.hpp"
#include "components/RestrictedBuffer.hpp"
#include "include/default_params.h"

namespace PV {

/**
 * The activity component for LCA layers with momentum.
 */
class MomentumLCAActivityComponent : public HyPerActivityComponent {

  public:
   MomentumLCAActivityComponent(char const *name, PVParams *parameters, Communicator *comm);

   virtual ~MomentumLCAActivityComponent();

  protected:
   MomentumLCAActivityComponent() {}

   void initialize(char const *name, PVParams *parameters, Communicator *comm);

   virtual void setObjectType();

   virtual void createComponentTable(char const *tableDescription) override;

   virtual ActivityBuffer *createActivity() override;

   virtual InternalStateBuffer *createInternalState() override;

   virtual RestrictedBuffer *createPrevDrive();

   Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual Response::Status allocateDataStructures() override;

   virtual Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;

   virtual Response::Status
   initializeState(std::shared_ptr<InitializeStateMessage const> message) override;

  protected:
   RestrictedBuffer *mPrevDrive = nullptr;
};

} // namespace PV

#endif // MOMENTUMLCAACTIVITYCOMPONENT_HPP_
