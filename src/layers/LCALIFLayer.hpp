/*
 * LCALIFLayer.hpp
 *
 *  Created on: Oct 3, 2012
 *      Author: slundquist
 */

#ifndef LCALIFLAYER_HPP_
#define LCALIFLAYER_HPP_

#include "HyPerLayer.hpp"
#include "LIFGap.hpp"

#define DEFAULT_DYNVTHSCALE 1.0

namespace PV {
class LCALIFLayer : public PV::LIFGap {
public:
   LCALIFLayer(const char* name, HyPerCol * hc); // The constructor called by other methods
   virtual ~LCALIFLayer();
   int updateState(float timef, float dt);
   int updateState(float timef, float dt, const PVLayerLoc * loc, pvdata_t * A, pvdata_t * V, int num_channels, pvdata_t * gSynHead, bool spiking, unsigned int * active_indices, unsigned int * num_active);
   int findFlag(int numMatrixCol, int numMatrixRow);

   virtual int checkpointRead(const char * cpDir, float * timef);
   virtual int checkpointWrite(const char * cpDir);

   inline float getTargetRate() {return targetRateHz;}
   const float * getVadpt() {return Vadpt;}
   const pvdata_t * getIntegratedSpikeCount() {return integratedSpikeCount;}
protected:
#ifdef PV_USE_OPENCL
   virtual int initializeThreadBuffers(const char * kernelName);
   virtual int initializeThreadKernels(const char * kernelName);

   // OpenCL buffers
   //
   CLBuffer * clG_Gap;
   CLBuffer * clGSynGap;

   virtual int getEVGSynGap() {return EV_LIFGAP_GSYN_GAP;}
   //virtual int getEVActivity() {return EV_LIFGAP_ACTIVITY;}
   virtual inline int getGSynEvent(ChannelType ch) {
      if(LIF::getGSynEvent(ch)>=0) return LIF::getGSynEvent(ch);
      if(ch==CHANNEL_GAP) return getEVGSynGap();
      return -1;
   }
   virtual int getNumCLEvents(){return NUM_LIFGAP_EVENTS;}
   virtual const char * getKernelName() {return "LCALIF_update_state";}
#endif

   int allocateBuffers();
   pvdata_t * integratedSpikeCount;      // plasticity decrement variable for postsynaptic layer
   float tauTHR;
   float targetRateHz;
   float Vscale;
   float * Vadpt;
   LCALIFLayer();
   int initialize(const char * name, HyPerCol * hc, int num_channels, const char * kernel_name);
   int initialize_base();
  // other methods and member variables
private:
  // other methods and member variables
};
}




#endif /* LCALIFLAYER_HPP_ */
