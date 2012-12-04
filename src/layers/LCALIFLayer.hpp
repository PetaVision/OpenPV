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
   int updateState(double timef, double dt);
   int updateState(double timef, double dt, const PVLayerLoc * loc, pvdata_t * A, pvdata_t * V, int num_channels, pvdata_t * gSynHead, bool spiking, unsigned int * active_indices, unsigned int * num_active);
   int findFlag(int numMatrixCol, int numMatrixRow);

   virtual int checkpointRead(const char * cpDir, double * timef);
   virtual int checkpointWrite(const char * cpDir);

   inline float getTargetRate() {return targetRateHz;}
   const float * getVadpt() {return Vadpt;}
   const pvdata_t * getIntegratedSpikeCount() {return integratedSpikeCount;}
   const pvdata_t * getVattained() {return Vattained;}
   const pvdata_t * getVmeminf() {return Vmeminf;}
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
   pvdata_t * G_Norm;                    // Copy of GSyn[CHANNEL_NORM] to be written out during checkpointing
   pvdata_t * GSynExcEffective;         // What is used as GSynExc, after normalizing, stored for checkpointing
   float tauTHR;
   float targetRateHz;
   float Vscale;
   float * Vadpt;
   float * Vattained; // Membrane potential before testing to see if a spike resets it to resting potential.  Output in checkpoints for diagnostic purposes but not otherwise used.
   float * Vmeminf;  // Asymptotic value of the membrane potential.  Output in checkpoints for diagnostic purposes but not otherwise used.
   bool normalizeInputFlag;
   LCALIFLayer();
   int initialize(const char * name, HyPerCol * hc, int num_channels, const char * kernel_name);
   int initialize_base();
  // other methods and member variables
private:
  // other methods and member variables
};
}




#endif /* LCALIFLAYER_HPP_ */
