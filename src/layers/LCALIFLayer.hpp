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
   virtual int allocateDataStructures();
   virtual int updateState(double timef, double dt);
//   int updateState(double timef, double dt, const PVLayerLoc * loc, pvdata_t * A, pvdata_t * V, int num_channels, pvdata_t * gSynHead, bool spiking, unsigned int * active_indices, unsigned int * num_active);
   int findFlag(int numMatrixCol, int numMatrixRow);

   virtual int checkpointWrite(const char * cpDir);

   inline float getTargetRate() {return targetRateHz;}
   const float * getVadpt() {return Vadpt;}
   const pvdata_t * getIntegratedSpikeCount() {return integratedSpikeCount;}
   const pvdata_t * getVattained() {return Vattained;}
   const pvdata_t * getVmeminf() {return Vmeminf;}
protected:
   LCALIFLayer();
   int initialize(const char * name, HyPerCol * hc, const char * kernel_name);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_tauTHR(enum ParamsIOFlag ioFlag);
   virtual void ioParam_targetRate(enum ParamsIOFlag ioFlag);
   virtual void ioParam_normalizeInput(enum ParamsIOFlag ioFlag);
   virtual void ioParam_Vscale(enum ParamsIOFlag ioFlag);
//#ifdef PV_USE_OPENCL
//
//   // OpenCL buffers
//   //
//   CLBuffer * clG_Gap;
//   CLBuffer * clGSynGap;
//
//   virtual int getEVGSynGap() {return EV_LIFGAP_GSYN_GAP;}
//   //virtual int getEVActivity() {return EV_LIFGAP_ACTIVITY;}
//   virtual inline int getGSynEvent(ChannelType ch) {
//      if(LIF::getGSynEvent(ch)>=0) return LIF::getGSynEvent(ch);
//      if(ch==CHANNEL_GAP) return getEVGSynGap();
//      return -1;
//   }
//   virtual int getNumCLEvents(){return NUM_LIFGAP_EVENTS;}
//   virtual const char * getKernelName() {return "LCALIF_update_state";}
//#endif
   virtual int readStateFromCheckpoint(const char * cpDir, double * timeptr);
   virtual int read_integratedSpikeCountFromCheckpoint(const char * cpDir, double * timeptr);
   virtual int readVadptFromCheckpoint(const char * cpDir, double * timeptr);

   int allocateBuffers();

   pvdata_t * integratedSpikeCount;      // plasticity decrement variable for postsynaptic layer
   pvdata_t * G_Norm;                    // Copy of GSyn[CHANNEL_NORM] to be written out during checkpointing
   pvdata_t * GSynExcEffective;         // What is used as GSynExc, after normalizing, stored for checkpointing
   pvdata_t * GSynInhEffective;         // What is used as GSynInh
   pvdata_t * excitatoryNoise;
   pvdata_t * inhibitoryNoise;
   pvdata_t * inhibNoiseB;
   float tauTHR;
   float targetRateHz;
   float Vscale;
   float * Vadpt;
   float * Vattained; // Membrane potential before testing to see if a spike resets it to resting potential.  Output in checkpoints for diagnostic purposes but not otherwise used.
   float * Vmeminf;  // Asymptotic value of the membrane potential.  Output in checkpoints for diagnostic purposes but not otherwise used.
   bool normalizeInputFlag;
  // other methods and member variables
private:
   int initialize_base();
  // other methods and member variables
}; // class LCALIFLayer

BaseObject * createLCALIFLayer(char const * name, HyPerCol * hc);

}  // namespace PV




#endif /* LCALIFLAYER_HPP_ */
