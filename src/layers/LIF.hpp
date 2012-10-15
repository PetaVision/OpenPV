/*
 * LIF.hpp
 *
 *  Created on: Dec 21, 2010
 *      Author: Craig Rasmussen
 *
 */

#ifndef LIF_HPP_
#define LIF_HPP_

#include "HyPerLayer.hpp"
#include "../kernels/LIF_params.h"
#include "../arch/opencl/pv_uint4.h"

#ifdef PV_USE_OPENCL
#include "../arch/opencl/CLBuffer.hpp"
#endif

#define NUM_LIF_EVENTS   4
//#define EV_LIF_GSYN_E     0
//#define EV_LIF_GSYN_I     1
#define EV_LIF_GSYN_IB    2
//#define EV_LIF_ACTIVITY  3

namespace PV
{

class LIF: public PV::HyPerLayer
{
public:

   friend int test_kernels(int argc, char * argv[]);
   friend int test_LIF    (int argc, char * argv[]);

   LIF(const char* name, HyPerCol * hc);
   LIF(const char* name, HyPerCol * hc, PVLayerType type);
   LIF(const char* name, HyPerCol * hc, PVLayerType type, int num_channels);
   virtual ~LIF();

   virtual int triggerReceive(InterColComm* comm);
   virtual int updateState(float time, float dt);
   virtual int updateStateOpenCL(float time, float dt);
   virtual int waitOnPublish(InterColComm* comm);
   virtual int setActivity();
   
   virtual int checkpointRead(const char * cpDir, float * timef);
   virtual int checkpointWrite(const char * cpDir);

#ifdef OBSOLETE // Marked obsolete July 13, 2012.  Restarting from last now handled by a call to checkpointRead from within HyPerLayer::initializeState
   virtual int readState (float * time);
#endif // OBSOLETE
#ifdef OBSOLETE // Marked obsolete Jul 13, 2012.  Dumping the state is now done by CheckpointWrite.
   virtual int writeState(float time, bool last=false);
#endif // OBSOLETE

   pvdata_t * getVth()              {return Vth;}
   virtual pvdata_t * getConductance(ChannelType ch) {
         return ch < this->numChannels ? G_E + ch*getNumNeurons() : NULL;
      }
/*
     pvdata_t * conductance = NULL;
      switch (ch) {
         case CHANNEL_EXC:  conductance = G_E; break;
         case CHANNEL_INH:  conductance = G_I; break;
         case CHANNEL_INHB: conductance = G_IB; break;
      }
      return conductance;
   }
*/

   virtual LIF_params * getLIFParams() {return &lParams;};

   int setParams(PVParams * p, float dt);

protected:

   // spikingFlag is member variable of HyPerLayer
   // bool spikingFlag;    // specifies that layer is spiking
   LIF_params lParams;
   uint4 * rand_state;  // state for random numbers

   pvdata_t * Vth;      // threshold potential
//   pvdata_t   VthRest;  // VthRest potential
   pvdata_t * G_E;      // excitatory conductance
   pvdata_t * G_I;      // inhibitory conductance
   pvdata_t * G_IB;

   char method;

#ifdef PV_USE_OPENCL
   virtual int initializeThreadBuffers(const char * kernelName);
   virtual int initializeThreadKernels(const char * kernelName);

   virtual int getNumCLEvents() {return NUM_LIF_EVENTS;}
   virtual const char * getKernelName() {return "LIF_update_state";}

   virtual int getEVGSynIB() {return EV_LIF_GSYN_IB;}
   //virtual int getEVActivity() {return EV_LIF_ACTIVITY;}
   virtual inline int getGSynEvent(ChannelType ch) {
      if(HyPerLayer::getGSynEvent(ch)>=0) return HyPerLayer::getGSynEvent(ch);
      if(ch==CHANNEL_INHB) return getEVGSynIB();
      return -1;
   }

   // OpenCL buffers
   //
   CLBuffer * clRand;
   CLBuffer * clVth;
   CLBuffer * clG_E;
   CLBuffer * clG_I;
   CLBuffer * clG_IB;
#endif

protected:
   LIF();
   int initialize(const char * name, HyPerCol * hc, PVLayerType type, int num_channels, const char * kernel_name);
   virtual int allocateBuffers();

private:
   int initialize_base();
   int findPostSynaptic(int dim, int maxSize, int col,
   // input: which layer, which neuron
   HyPerLayer *lSource, float pos[],

   // output: how many of our neurons are connected.
   // an array with their indices.
   // an array with their feature vectors.
   int* nNeurons, int nConnectedNeurons[], float *vPos);
};

} // namespace PV

#endif /* LIF_HPP_ */
