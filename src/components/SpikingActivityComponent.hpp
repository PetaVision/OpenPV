/*
 * SpikingActivityComponent.hpp
 *
 *  Created on: Sep 12, 2018
 *      Author: twatkins
 */

#ifndef SPIKINGACTIVITYCOMPONENT_HPP_
#define SPIKINGACTIVITYCOMPONENT_HPP_

#include "components/HyPerActivityComponent.hpp"

typedef PV::HyPerActivityComponent<PV::GSynAccumulator, PV::InternalStateBuffer, PV::ActivityBuffer>
      SpikingActivityComponentBase;

namespace PV {

/**
 * An activity component that models a leaky membrane potential.
 * The layer numerically integrates dV/dt = -1/tau * (V - gSynInput ),
 * where tau is given by the integrationTime parameter.
 * If V is <= VThresh, the activity is set to zero.
 * If V is greater than VThresh, V resets to zero and the activity is set to one.
 */
class SpikingActivityComponent : public SpikingActivityComponentBase {
  protected:
   /**
    * @brief integrationTime: The parameter governing the decay rate of the internal state.
    * The default is infinity.
    * @details: technically, the default is FLT_MAX, usually approx. 3.4*10^38 for
    * 32-bit float precision.
    */
   virtual void ioParam_integrationTime(enum ParamsIOFlag ioFlag);

   /**
    * @brief VThresh: The threshold value of V at which the neuron spikes, resetting V to zero
    * and setting A to 1.
    */
   virtual void ioParam_VThresh(enum ParamsIOFlag ioFlag);

  public:
   SpikingActivityComponent(char const *name, PVParams *params, Communicator const *comm);

   virtual ~SpikingActivityComponent();

  protected:
   SpikingActivityComponent() {}

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual void setObjectType() override;

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual Response::Status updateActivity(double simTime, double deltaTime) override;

   // Data Members
  protected:
   float mVThresh; // no default; required parameter
   float mIntegrationTime = FLT_MAX;
};

} // namespace PV

#endif // SPIKINGACTIVITYCOMPONENT_HPP_
