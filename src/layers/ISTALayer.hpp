/*
 * ISTALayer.hpp
 *
 *  Created on: Jan 24, 2013
 *      Author: garkenyon
 */

#ifndef ISTALAYER_HPP_
#define ISTALAYER_HPP_
// TODO: Take care of code duplication between ISTALayer and HyPerLCALayer.

#include <io/AdaptiveTimeScaleProbe.hpp>
#include "ANNLayer.hpp"

namespace PV {

class ISTALayer: public PV::ANNLayer {
public:
   ISTALayer(const char * name, HyPerCol * hc);
   virtual ~ISTALayer();
   virtual double getDeltaUpdateTime();
   virtual int requireChannel(int channelNeeded, int * numChannelsResult) ;

protected:
   ISTALayer();
   int initialize(const char * name, HyPerCol * hc);
   virtual int allocateDataStructures();
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);

   /**
    * List of parameters needed from the ISTALayer class
    * @name HyPerConn Parameters
    * @{
    */

   /**
    * timeConstantTau: the time constant tau for the LCA dynamics, which models the equation dV/dt = 1/tau*(-V+s*A+GSyn)
    */
   virtual void ioParam_timeConstantTau(enum ParamsIOFlag ioFlag);
   /**
    * timeConstantTau: the self-interaction coefficient s for the LCA dynamics, which models the equation dV/dt = 1/tau*(-V+s*A+GSyn)
    */
   virtual void ioParam_selfInteract(enum ParamsIOFlag ioFlag);

   /**
    * @brief adaptiveTimeScaleProbe: If using adaptive timesteps, the name of the AdaptiveTimeScaleProbe that will compute the dt values.
    */
   virtual void ioParam_adaptiveTimeScaleProbe(enum ParamsIOFlag ioFlag);
   /** @} */

   virtual int updateState(double time, double dt);

#ifdef PV_USE_CUDA
   virtual int updateStateGpu(double time, double dt);
#endif

   virtual float getChannelTimeConst(enum ChannelType channel_type){return timeConstantTau;};

#ifdef PV_USE_CUDA
   virtual int allocateUpdateKernel();
#endif

   double * deltaTimes(); // TODO: make const-correct
   // Better name?  getDeltaTimes isn't good because it sounds like it's just the getter-method.

private:
   int initialize_base();
#ifdef PV_USE_CUDA
   PVCuda::CudaBuffer* d_dtAdapt;
#endif

// Data members
protected:
   pvdata_t timeConstantTau;
   bool selfInteract;
   char * mAdaptiveTimeScaleProbeName = nullptr;
   AdaptiveTimeScaleProbe * mAdaptiveTimeScaleProbe = nullptr;
   std::vector<double> mDeltaTimes;
}; // class ISTALayer

} /* namespace PV */
#endif /* ISTALAYER_HPP_ */
