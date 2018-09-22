/*
 * HyPerLCALayer.hpp
 *
 *  Created on: Jan 24, 2013
 *      Author: garkenyon
 */

#ifndef HYPERLCALAYER_HPP_
#define HYPERLCALAYER_HPP_

#include "ANNLayer.hpp"
#include "probes/AdaptiveTimeScaleProbe.hpp"

namespace PV {

class HyPerLCALayer : public PV::ANNLayer {
  public:
   HyPerLCALayer(const char *name, HyPerCol *hc);
   virtual ~HyPerLCALayer();
   virtual double getDeltaUpdateTime() const override;

  protected:
   HyPerLCALayer();
   int initialize(const char *name, HyPerCol *hc);
   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   virtual Response::Status allocateDataStructures() override;
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   /**
    * List of parameters needed from the HyPerLCALayer class
    * @name HyPerLCALayer Parameters
    * @{
    */

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

   virtual LayerInputBuffer *createLayerInput() override;

   virtual Response::Status
   initializeState(std::shared_ptr<InitializeStateMessage const> message) override;

   virtual Response::Status updateState(double time, double dt) override;

#ifdef PV_USE_CUDA
   virtual Response::Status copyInitialStateToGPU() override;

   virtual Response::Status updateStateGpu(double time, double dt) override;

   virtual int allocateUpdateKernel() override;
#endif

   double *deltaTimes(); // TODO: make const-correct
   // Better name?  getDeltaTimes isn't good because it sounds like it's just the getter-method.

  private:
   int initialize_base();

   // Data members
  protected:
   float scaledTimeConstantTau = 1.0f; // The tau from the TauLayerInputBuffer, divided by dt
   bool selfInteract;
   char *mAdaptiveTimeScaleProbeName               = nullptr;
   AdaptiveTimeScaleProbe *mAdaptiveTimeScaleProbe = nullptr;
   std::vector<double> mDeltaTimes;

#ifdef PV_USE_CUDA
   PVCuda::CudaBuffer *d_dtAdapt;
   PVCuda::CudaBuffer *d_verticesV;
   PVCuda::CudaBuffer *d_verticesA;
   PVCuda::CudaBuffer *d_slopes;
#endif
}; // class HyPerLCALayer

} /* namespace PV */
#endif /* HYPERLCALAYER_HPP_ */
