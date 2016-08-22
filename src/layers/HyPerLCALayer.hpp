/*
 * HyPerLCALayer.hpp
 *
 *  Created on: Jan 24, 2013
 *      Author: garkenyon
 */

#ifndef HYPERLCALAYER_HPP_
#define HYPERLCALAYER_HPP_

#include <io/AdaptiveTimestepProbe.hpp>
#include "ANNLayer.hpp"

namespace PV {

class HyPerLCALayer: public PV::ANNLayer {
public:
   HyPerLCALayer(const char * name, HyPerCol * hc);
   virtual ~HyPerLCALayer();
   virtual double getDeltaUpdateTime();
   virtual int requireChannel(int channelNeeded, int * numChannelsResult) ;

protected:
   HyPerLCALayer();
   int initialize(const char * name, HyPerCol * hc);
   virtual int communicateInitInfo() override;
   virtual int allocateDataStructures() override;
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);

   /**
    * List of parameters needed from the HyPerLCALayer class
    * @name HyPerConn Parameters
    * @{
    */

#ifdef OBSOLETE // Marked obsolete June 27, 2016.
   /**
    * numChannels: obsolete parameter, as the layer infers the number of channels from connections connecting to it.
    */
   virtual void ioParam_numChannels(enum ParamsIOFlag ioFlag); // numChannels as a HyPerLayer parameter is obsolete but generates warning.  After a suitable fade time, ioParam_numChannels can be removed
#endif // OBSOLETE // Marked obsolete June 27, 2016.

   /**
    * @brief timeConstantTau: the time constant tau for the LCA dynamics, which models the equation dV/dt = 1/tau*(-V+s*A+GSyn)
    */
   virtual void ioParam_timeConstantTau(enum ParamsIOFlag ioFlag);

   /**
    * @brief timeConstantTau: the self-interaction coefficient s for the LCA dynamics, which models the equation dV/dt = 1/tau*(-V+s*A+GSyn)
    */
   virtual void ioParam_selfInteract(enum ParamsIOFlag ioFlag);

   /**
    * @brief adaptiveTimestepProbe: If using adaptive timesteps, the name of the AdaptiveTimestepProbe that will compute the dt values.
    */
   virtual void ioParam_adaptiveTimestepProbe(enum ParamsIOFlag ioFlag);
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

// Data members
protected:
   pvdata_t timeConstantTau;
   bool selfInteract;
   char * mAdaptiveTimestepProbeName = nullptr;
   AdaptiveTimestepProbe * mAdaptiveTimestepProbe = nullptr;
   std::vector<double> mDeltaTimes;

#ifdef PV_USE_CUDA
   PVCuda::CudaBuffer* d_dtAdapt;
   PVCuda::CudaBuffer* d_verticesV;
   PVCuda::CudaBuffer* d_verticesA;
   PVCuda::CudaBuffer* d_slopes;
#endif
}; // class HyPerLCALayer

} /* namespace PV */
#endif /* HYPERLCALAYER_HPP_ */
