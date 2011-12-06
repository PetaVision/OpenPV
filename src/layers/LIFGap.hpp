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
#define EV_LIF_GSYN_GAP     NUM_LIF_EVENTS + 1


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

   virtual int checkpointRead(float * timef);
   virtual int checkpointWrite();

   int virtual readState(float * time);
   int virtual writeState(float time, bool last);

protected:

   LIFGap();
   int initialize(const char * name, HyPerCol * hc, PVLayerType type, int num_channels, const char * kernel_name);

   pvdata_t * G_Gap;
   pvdata_t sumGap;

#ifdef PV_USE_OPENCL
   virtual int initializeThreadBuffers(char * kernelName);
   virtual int initializeThreadKernels(char * kernelName);

   // OpenCL buffers
   //
   CLBuffer * clG_Gap;
   CLBuffer * clGSynGap;

   virtual int getNumCLEvents(){return NUM_LIFGAP_EVENTS;}
#endif

private:
   int initialize_base();

};

} /* namespace PV */
#endif /* LIFGAP_HPP_ */
