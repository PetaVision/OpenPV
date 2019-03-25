/*
 * HyPerLCAInternalStateBuffer.hpp
 *
 *  Created on: Jan 24, 2013
 *      Author: garkenyon
 */

#ifndef HYPERLCAINTERNALSTATEBUFFER_HPP_
#define HYPERLCAINTERNALSTATEBUFFER_HPP_

#include "components/HyPerInternalStateBuffer.hpp"

#include "probes/AdaptiveTimeScaleProbe.hpp"

namespace PV {

class HyPerLCAInternalStateBuffer : public HyPerInternalStateBuffer {
  protected:
   /**
    * List of parameters needed from the HyPerLCAInternalStateBuffer class
    * @name HyPerLCAInternalStateBuffer Parameters
    * @{
    */

   /**
    * @brief timeConstantTau: the time constant tau,
    * used in solving the differential equation dV/dt = 1/tau * (-V + A + GSyn).
    */
   virtual void ioParam_timeConstantTau(enum ParamsIOFlag ioFlag);

   /**
    * @brief selfInteract: the self-interaction coefficient s for the LCA dynamics, which models
    * the equation dV/dt = 1/tau*(-V+s*A+GSyn)
    */
   virtual void ioParam_selfInteract(enum ParamsIOFlag ioFlag);

   /**
    * @brief adaptiveTimeScaleProbe: If using adaptive timesteps, the name of the
    * AdaptiveTimeScaleProbe that will compute the dt values.
    */
   virtual void ioParam_adaptiveTimeScaleProbe(enum ParamsIOFlag ioFlag);
   /** @} */

  public:
   HyPerLCAInternalStateBuffer(const char *name, PVParams *params, Communicator const *comm);
   virtual ~HyPerLCAInternalStateBuffer();

  protected:
   HyPerLCAInternalStateBuffer();
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

   void runKernel();
#endif // PV_USE_CUDA

   double const *deltaTimes(double simTime, double deltaTime);
   // Better name?  getDeltaTimes isn't good because it sounds like it's just the getter-method.

   // Data members
  protected:
   double mTimeConstantTau = 1.0; // The time constant tau in the equation dV/dt=1/tau*(-V+A+GSyn).
   double mScaledTimeConstantTau = 1.0; // tau/dt, used in numerical integration.
   bool mSelfInteract;
   char *mAdaptiveTimeScaleProbeName               = nullptr;
   AdaptiveTimeScaleProbe *mAdaptiveTimeScaleProbe = nullptr;
   std::vector<double> mDeltaTimes;
   ActivityBuffer *mActivity = nullptr;
#ifdef PV_USE_CUDA
   PVCuda::CudaBuffer *mCudaDtAdapt = nullptr;
#endif
}; // class HyPerLCAInternalStateBuffer

} /* namespace PV */
#endif /* HYPERLCAINTERNALSTATEBUFFER_HPP_ */
