/*
 * LeakyInternalStateBuffer.hpp
 *
 *  Created on: Sep 6, 2018
 *      Author: Pete Schultz
 */

#ifndef LEAKYINTERNALSTATEBUFFER_HPP_
#define LEAKYINTERNALSTATEBUFFER_HPP_

#include "components/InternalStateBuffer.hpp"

namespace PV {

/**
 * A membrane potential component that models a leaky membrane potential.
 * The layer numerically integrates dV/dt = -V/tau + gSynExc - gSynInh,
 * where tau is given by the integrationTime parameter.
 * tau is in the same units that the HyPerCol dt parameter is in.
 */
class LeakyInternalStateBuffer : public InternalStateBuffer {
  protected:
   /**
    * List of parameters used by the ANNErrorLayer class
    * @name ANNErrorLayer Parameters
    * @{
    */

   /**
    * @brief: integrationTime:
    * The time constant for the decay ("leakiness") of the membrane potential
    */
   virtual void ioParam_integrationTime(enum ParamsIOFlag ioFlag);

  public:
   LeakyInternalStateBuffer(char const *name, HyPerCol *hc);

   virtual ~LeakyInternalStateBuffer();

   virtual void updateBuffer(double simTime, double deltaTime) override;

  protected:
   LeakyInternalStateBuffer() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual void setObjectType() override;

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

  protected:
   float mIntegrationTime = FLT_MAX;
};

} // namespace PV

#endif // LEAKYINTERNALSTATEBUFFER_HPP_
