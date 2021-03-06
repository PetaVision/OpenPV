/*
 * ISTAInternalStateBuffer.hpp
 *
 *  Created on: Jan 24, 2013
 *      Author: garkenyon
 */

#ifndef ISTAINTERNALSTATEBUFFER_HPP_
#define ISTAINTERNALSTATEBUFFER_HPP_

#include "components/HyPerInternalStateBuffer.hpp"

#include "components/ANNActivityBuffer.hpp"
#include "probes/AdaptiveTimeScaleProbe.hpp"

namespace PV {

class ISTAInternalStateBuffer : public HyPerInternalStateBuffer {
  protected:
   /**
    * List of parameters needed from the ISTAInternalStateBuffer class
    * @name ISTAInternalStateBuffer Parameters
    * @{
    */

   /**
    * @brief timeConstantTau: the time constant tau,
    * used in solving the differential equation dV/dt = 1/tau * (-V + A + GSyn).
    */
   virtual void ioParam_timeConstantTau(enum ParamsIOFlag ioFlag);
   /**
    * @brief adaptiveTimeScaleProbe: If using adaptive timesteps, the name of the
    * AdaptiveTimeScaleProbe that will compute the dt values.
    */
   virtual void ioParam_adaptiveTimeScaleProbe(enum ParamsIOFlag ioFlag);
   /** @} */

  public:
   ISTAInternalStateBuffer(const char *name, PVParams *params, Communicator const *comm);
   virtual ~ISTAInternalStateBuffer();

  protected:
   ISTAInternalStateBuffer();
   void initialize(const char *name, PVParams *params, Communicator const *comm);
   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   virtual Response::Status allocateDataStructures() override;
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual Response::Status
   initializeState(std::shared_ptr<InitializeStateMessage const> message) override;

#ifdef PV_USE_CUDA
   virtual void allocateUpdateKernel() override;
#endif

   virtual void updateBufferCPU(double simTime, double deltaTime) override;
#ifdef PV_USE_CUDA
   virtual void updateBufferGPU(double simTime, double deltaTime) override;

   virtual void runKernel();
#endif // PV_USE_CUDA

   double const *deltaTimes(double simTime, double deltaTime);
   // Better name?  getDeltaTimes isn't good because it sounds like it's just the getter-method.

   // Data members
  protected:
   double mTimeConstantTau = 1.0; // The time constant tau in the equation dV/dt=1/tau*(-V+A+GSyn).
   float mScaledTimeConstantTau                    = 1.0f; // tau/dt, used in numerical integration.
   char *mAdaptiveTimeScaleProbeName               = nullptr;
   AdaptiveTimeScaleProbe *mAdaptiveTimeScaleProbe = nullptr;
   std::vector<double> mDeltaTimes;
   ANNActivityBuffer *mActivity = nullptr;
#ifdef PV_USE_CUDA
   PVCuda::CudaBuffer *mCudaDtAdapt = nullptr;
#endif
}; // class ISTAInternalStateBuffer

} /* namespace PV */
#endif /* ISTAINTERNALSTATEBUFFER_HPP_ */
