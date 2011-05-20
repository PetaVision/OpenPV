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
   friend int test_LIF    (int argc, char * argv[]);

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
      pvdata_t * conductance;
      switch (ch) {
         case CHANNEL_EXC:  conductance = G_E; break;
         case CHANNEL_INH:  conductance = G_I; break;
         case CHANNEL_INHB: conductance = G_IB; break;
      }
      return conductance;
   }

   virtual LIF_params * getLIFParams() {return &lParams;};
   virtual pvdata_t * getWmax() {return Wmax;}
   virtual pvdata_t * getVthRest(){return VthRest;}
   virtual pvdata_t * getR(){return R;}
   virtual bool getLocalWmaxFlag() {return localWmaxFlag;}
   virtual bool getLocalVthRestFlag() {return localVthRestFlag;}

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

   bool     localWmaxFlag;  // presence of rate dependent wMax;
   bool     localVthRestFlag; // presence of rate dependent VthRest
   pvdata_t * Wmax;   // adaptive upper STDP weight boundary
   pvdata_t * VthRest; // adaptive VthRest
   float    tauVthRest;
   float    alphaVthRest;
   float    wMax;
   float    wMin;
   float    alphaW;   // params in Wmax dynamics.
   float    tauWmax;  // should include in LIF_params
   float    averageR; // predefined average rate

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
