/*
 * MomentumLCAActivityComponent.hpp
 *
 *  Created on: Mar 15, 2016
 *      Author: slundquist
 */

#ifndef MOMENTUMLCAACTIVITYCOMPONENT_HPP_
#define MOMENTUMLCAACTIVITYCOMPONENT_HPP_

#include "components/HyPerActivityComponent.hpp"

#include "components/ANNActivityBuffer.hpp"
#include "components/GSynAccumulator.hpp"
#include "components/MomentumLCAInternalStateBuffer.hpp"
#include "components/RestrictedBuffer.hpp"

namespace PV {

typedef HyPerActivityComponent<GSynAccumulator, MomentumLCAInternalStateBuffer, ANNActivityBuffer>
      BaseMomentumActivityComponent;

/**
 * The activity component for LCA layers with momentum.
 */
class MomentumLCAActivityComponent : public BaseMomentumActivityComponent {

  public:
   MomentumLCAActivityComponent(char const *name, PVParams *parameters, Communicator const *comm);

   virtual ~MomentumLCAActivityComponent();

  protected:
   MomentumLCAActivityComponent() {}

   void initialize(char const *name, PVParams *parameters, Communicator const *comm);

   virtual void setObjectType() override;

   virtual void createComponentTable(char const *tableDescription) override;

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
