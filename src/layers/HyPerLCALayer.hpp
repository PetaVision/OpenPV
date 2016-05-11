/*
 * HyPerLCALayer.hpp
 *
 *  Created on: Jan 24, 2013
 *      Author: garkenyon
 */

#ifndef HYPERLCALAYER_HPP_
#define HYPERLCALAYER_HPP_

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
   virtual int allocateDataStructures();
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);

   /**
    * List of parameters needed from the HyPerLCALayer class
    * @name HyPerConn Parameters
    * @{
    */

   /**
    * numChannels: obsolete parameter, as the layer infers the number of channels from connections connecting to it.
    */
   virtual void ioParam_numChannels(enum ParamsIOFlag ioFlag); // numChannels as a HyPerLayer parameter is obsolete but generates warning.  After a suitable fade time, ioParam_numChannels can be removed

   /**
    * timeConstantTau: the time constant tau for the LCA dynamics, which models the equation dV/dt = 1/tau*(-V+s*A+GSyn)
    */
   virtual void ioParam_timeConstantTau(enum ParamsIOFlag ioFlag);
#ifdef OBSOLETE // Marked obsolete Jul 9, 2015.  None of these member variables are being used.
   virtual void ioParam_numWindowX(enum ParamsIOFlag ioFlag);
   virtual void ioParam_numWindowY(enum ParamsIOFlag ioFlag);
   virtual void ioParam_windowSymX(enum ParamsIOFlag ioFlag);
   virtual void ioParam_windowSymY(enum ParamsIOFlag ioFlag);
#endif // OBSOLETE // Marked obsolete Jul 9, 2015.  None of these member variables are being used.
   /**
    * timeConstantTau: the self-interaction coefficient s for the LCA dynamics, which models the equation dV/dt = 1/tau*(-V+s*A+GSyn)
    */
   virtual void ioParam_selfInteract(enum ParamsIOFlag ioFlag);

   /** @} */

   virtual int doUpdateState(double time, double dt, const PVLayerLoc * loc, pvdata_t * A,
         pvdata_t * V, int num_channels, pvdata_t * gSynHead);

#ifdef PV_USE_CUDA
   virtual int doUpdateStateGpu(double time, double dt, const PVLayerLoc * loc, pvdata_t * A,
      pvdata_t * V, int num_channels, pvdata_t * gSynHead);
#endif

   virtual float getChannelTimeConst(enum ChannelType channel_type){return timeConstantTau;};

#if defined(PV_USE_OPENCL) || defined(PV_USE_CUDA)
   virtual int allocateUpdateKernel();
#endif


   pvdata_t timeConstantTau;
   bool selfInteract;

#ifdef PV_USE_CUDA
   PVCuda::CudaBuffer* d_dtAdapt;
   PVCuda::CudaBuffer* d_verticesV;
   PVCuda::CudaBuffer* d_verticesA;
   PVCuda::CudaBuffer* d_slopes;
#endif

private:
   int initialize_base();
}; // class HyPerLCALayer

BaseObject * createHyPerLCALayer(char const * name, HyPerCol * hc);

} /* namespace PV */
#endif /* HYPERLCALAYER_HPP_ */
