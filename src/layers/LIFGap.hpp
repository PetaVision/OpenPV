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
   LIFGap(const char* name, HyPerCol * hc, PVLayerType type);
   virtual ~LIFGap();

   void addGapStrength(float gap_strength){sumGap += gap_strength;}
   int virtual updateStateOpenCL(float time, float dt);
   int virtual triggerReceive(InterColComm* comm);
   int virtual updateState(float time, float dt);

   virtual int checkpointRead(const char * cpDir, float * timef);
   virtual int checkpointWrite(const char * cpDir);

#ifdef OBSOLETE // Marked obsolete July 13, 2012.  Restarting from last now handled by a call to checkpointRead from within HyPerLayer::initializeState
   int virtual readState(float * time);
#endif // OBSOLETE
#ifdef OBSOLETE // Marked obsolete Jul 13, 2012.  Dumping the state is now done by CheckpointWrite.
   int virtual writeState(float time, bool last);
#endif // OBSOLETE

protected:

   LIFGap();
   int initialize(const char * name, HyPerCol * hc, PVLayerType type, int num_channels, const char * kernel_name);
   virtual int allocateBuffers();

   pvdata_t * G_Gap;
   pvdata_t sumGap;
   // char method;  // Moved to LIF

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
   virtual const char * getKernelName() {return "LIFGap_update_state";}
#endif

private:
   int initialize_base();

};

} /* namespace PV */
#endif /* LIFGAP_HPP_ */
