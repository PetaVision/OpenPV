/*
 * SigmoidActivityBuffer.hpp
 *
 *  Created on: May 11, 2011
 *      Author: garkenyon
 */

#ifndef SIGMOIDACTIVITYBUFFER_HPP_
#define SIGMOIDACTIVITYBUFFER_HPP_

#include "components/VInputActivityBuffer.hpp"
#include "include/default_params.h"

namespace PV {

/**
 * An activity updater component to set the activity as a sigmoid of the membrane potential.
 */
class SigmoidActivityBuffer : public VInputActivityBuffer {
  protected:
   /**
    * List of parameters used by the SigmoidActivityBuffer class
    * @name ANNLayer Parameters
    * @{
    */

   /**
    * @brief Vrest:
    */
   virtual void ioParam_Vrest(ParamsIOSwitch ioSwitch);

   /**
    * @brief VthRest:
    */
   virtual void ioParam_VthRest(ParamsIOSwitch ioSwitch);

   /**
    * @brief InverseFlag: If true, the activity decreases from 1 to 0
    * as the membrane potential increases from -infinity to infinity.
    * If InverseFlag is false, the activity increases from 0 to 1.
    * Default is false.
    */
   virtual void ioParam_InverseFlag(ParamsIOSwitch ioSwitch);

   /**
    * @brief SigmoidFlag: If this flag is false, the activity is
    * a piecewise linear function of V. If the flag is true, the
    * activity is a true sigmoid function.
    * Default is true.
    */
   virtual void ioParam_SigmoidFlag(ParamsIOSwitch ioSwitch);

   /**
    * @brief SigmoidAlpha:
    */
   virtual void ioParam_SigmoidAlpha(ParamsIOSwitch ioSwitch);
   /** @} */
  public:
   SigmoidActivityBuffer(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);

   virtual ~SigmoidActivityBuffer();

  protected:
   SigmoidActivityBuffer() {}

   void initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);

   virtual void setObjectType() override;

   virtual int ioParamsFillGroup(ParamsIOSwitch ioSwitch) override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual Response::Status
   initializeState(std::shared_ptr<InitializeStateMessage const> message) override;

   /**
    * Computes A from V using the sigmoid function defined by the input parameters.
    */
   virtual void updateBufferCPU(double simTime, double deltaTime) override;

  protected:
   float mVrest        = (float)V_REST;
   float mVthRest      = (float)VTH_REST;
   bool mInverseFlag   = false;
   bool mSigmoidFlag   = true;
   float mSigmoidAlpha = 0.1;
};

} // namespace PV

#endif // SIGMOIDACTIVITYBUFFER_HPP_
