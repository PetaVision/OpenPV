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
#include "../arch/opencl/CLKernel.hpp"
#include "../arch/opencl/pv_uint4.h"
#include "../kernels/LIF_params.h"

#define NUM_LIF_EVENTS   4
#define EV_LIF_PHI_E     0
#define EV_LIF_PHI_I     1
#define EV_LIF_PHI_IB    2
#define EV_LIF_ACTIVITY  3

namespace PV
{

class LIF: public PV::HyPerLayer
{
public:

   friend int test_kernels(int argc, char * argv[]);

   LIF(const char* name, HyPerCol * hc);
   LIF(const char* name, HyPerCol * hc, PVLayerType type);
   virtual ~LIF();

   virtual int triggerReceive(InterColComm* comm);
   virtual int updateState(float time, float dt);
   virtual int updateStateOpenCL(float time, float dt);
   virtual int waitOnPublish(InterColComm* comm);
   
   virtual int readState (float * time);
   virtual int writeState(float time, bool last=false);
   
   pvdata_t * getAverageActivity()  {return R;}
   pvdata_t * getVth()              {return Vth;}
   pvdata_t * getConductance(ChannelType ch) {
      switch (ch) {
         case CHANNEL_EXC:  return G_E;
         case CHANNEL_INH:  return G_I;
         case CHANNEL_INHB: return G_IB;
      }
   }
   
   int setParams(PVParams * p);

protected:

   bool spikingFlag;    // specifies that layer is spiking
   LIF_params lParams;
   uint4 * rand_state;  // state for random numbers

   pvdata_t * Vth;      // threshhold potential
   pvdata_t * G_E;      // excitatory conductance
   pvdata_t * G_I;      // inhibitory conductance
   pvdata_t * G_IB;
   pvdata_t * R;        // average activity (not extended)

#ifdef PV_USE_OPENCL
   virtual int initializeThreadBuffers();
   virtual int initializeThreadKernels();

   // OpenCL buffers
   //
   CLBuffer * clRand;
   CLBuffer * clVth;
   CLBuffer * clG_E;
   CLBuffer * clG_I;
   CLBuffer * clG_IB;
   CLBuffer * clR;
#endif

private:
   virtual int initialize(PVLayerType type);

   int findPostSynaptic(int dim, int maxSize, int col,
	// input: which layer, which neuron
			HyPerLayer *lSource, float pos[],

			// output: how many of our neurons are connected.
			// an array with their indices.
			// an array with their feature vectors.
			int* nNeurons, int nConnectedNeurons[], float *vPos);

};

} // namespace PV

#ifdef __cplusplus
extern "C"
{
#endif


#ifdef __cplusplus
}
#endif

#endif /* LIF_HPP_ */
