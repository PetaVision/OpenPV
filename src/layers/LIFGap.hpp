/*
 * LIFGap.hpp
 *
 *  Created on: Jul 29, 2011
 *      Author: garkenyon
 */

#ifndef LIFGAP_HPP_
#define LIFGAP_HPP_

#include "LIF.hpp"

#define NUM_LIFGAP_EVENTS   1 + NUM_LIF_EVENTS  // ???
//#define EV_LIF_GSYN_GAP     NUM_LIF_EVENTS + 1
#define EV_LIFGAP_GSYN_GAP     3
//#define EV_LIFGAP_ACTIVITY  4


namespace PV {

class LIFGap: public PV::LIF {
public:
   LIFGap(const char* name, HyPerCol * hc);
   virtual ~LIFGap();

   int virtual updateStateOpenCL(double time, double dt);
   int virtual updateState(double time, double dt);

   int virtual checkpointWrite(const char * cpDir);
   int virtual readStateFromCheckpoint(const char * cpDir, double * timeptr);

   const pvgsyndata_t * getGapStrength() { return gapStrength; }

protected:

   LIFGap();
   int initialize(const char * name, HyPerCol * hc, const char * kernel_name);
   virtual int allocateConductances(int num_channels);
   virtual int readGapStrengthFromCheckpoint(const char * cpDir, double * timeptr);

//#ifdef PV_USE_OPENCL
//   virtual int initializeThreadBuffers(const char * kernelName);
//   virtual int initializeThreadKernels(const char * kernelName);
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
//   virtual const char * getKernelName() {return "LIFGap_update_state";}
//#endif

private:
   int initialize_base();
   pvgsyndata_t * gapStrength;
   bool gapStrengthInitialized;
   int calcGapStrength();

}; // class LIFGap

BaseObject * createLIFGap(char const * name, HyPerCol * hc);

} /* namespace PV */
#endif /* LIFGAP_HPP_ */
