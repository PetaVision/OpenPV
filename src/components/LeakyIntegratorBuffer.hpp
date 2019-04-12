/*
 * LeakyIntegratorBuffer.hpp
 *
 *  Created on: Feb 12, 2013
 *      Author: pschultz
 */

#ifndef LEAKYINTEGRATORBUFFER_HPP_
#define LEAKYINTEGRATORBUFFER_HPP_

#include "components/HyPerInternalStateBuffer.hpp"

namespace PV {

/**
 * A membrane potential component that models a leaky membrane potential.
 * The layer numerically integrates dV/dt = -V/tau + gSynExc - gSynInh,
 * where tau is given by the integrationTime parameter.
 * tau is in the same units that the HyPerCol dt parameter is in.
 */
class LeakyIntegratorBuffer : public HyPerInternalStateBuffer {
  protected:
   /**
    * List of parameters used by the LeakyIntegratorBuffer class
    * @name LeakyIntegrator Parameters
    * @{
    */

   /**
    * @brief: integrationTime:
    * The time constant for the decay ("leakiness") of the membrane potential
    */
   virtual void ioParam_integrationTime(enum ParamsIOFlag ioFlag);

  public:
   LeakyIntegratorBuffer(char const *name, PVParams *params, Communicator const *comm);

   virtual ~LeakyIntegratorBuffer();

  protected:
   LeakyIntegratorBuffer() {}

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual void setObjectType() override;

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual void updateBufferCPU(double simTime, double deltaTime) override;

  protected:
   float mIntegrationTime = FLT_MAX;
};

} // namespace PV

#endif // LEAKYINTEGRATORBUFFER_HPP_
