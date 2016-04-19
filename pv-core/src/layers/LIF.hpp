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
#include "../columns/Random.hpp"
#include "../kernels/LIF_params.h"

//#ifdef PV_USE_OPENCL
//#include "../arch/opencl/CLBuffer.hpp"
//#endif

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
   LIF(const char* name, HyPerCol * hc, int num_channels);
   virtual ~LIF();

   virtual int communicateInitInfo();
   virtual int allocateDataStructures();

   virtual int updateState(double time, double dt);
   virtual int updateStateOpenCL(double time, double dt);
   virtual int waitOnPublish(InterColComm* comm);
   virtual int setActivity();
   
   virtual int checkpointWrite(const char * cpDir);

   pvdata_t * getVth()              {return Vth;}
   virtual pvconductance_t * getConductance(ChannelType ch) {
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
   virtual float getChannelTimeConst(enum ChannelType channel_type);

   virtual LIF_params * getLIFParams() {return &lParams;};

   virtual bool activityIsSpiking() { return true; }


protected:
   LIF_params lParams;
   Random * randState;
//#ifdef PV_USE_OPENCL
//   //TODO-Rasmussen-2014.5.24 - need to figure out interaction between Random class and rand_state
//   taus_uint4 * rand_state;  // state for random numbers
//#endif

   pvdata_t * Vth;      // threshold potential
   pvconductance_t * G_E;      // excitatory conductance
   pvconductance_t * G_I;      // inhibitory conductance
   pvconductance_t * G_IB;

   char * methodString; // 'arma', 'before', or 'original'
   char method;         // 'a', 'b', or 'o', the first character of methodString

//#ifdef PV_USE_OPENCL
//   virtual int initializeThreadBuffers(const char * kernelName);
//   virtual int initializeThreadKernels(const char * kernelName);
//
//   virtual int getNumCLEvents() {return NUM_LIF_EVENTS;}
//   virtual const char * getKernelName() {return "LIF_update_state";}
//
//   virtual int getEVGSynIB() {return EV_LIF_GSYN_IB;}
//   //virtual int getEVActivity() {return EV_LIF_ACTIVITY;}
//   virtual inline int getGSynEvent(ChannelType ch) {
//      if(HyPerLayer::getGSynEvent(ch)>=0) return HyPerLayer::getGSynEvent(ch);
//      if(ch==CHANNEL_INHB) return getEVGSynIB();
//      return -1;
//   }
//
//   // OpenCL buffers
//   //
//   CLBuffer * clRand;
//   CLBuffer * clVth;
//   CLBuffer * clG_E;
//   CLBuffer * clG_I;
//   CLBuffer * clG_IB;
//#endif

protected:
   LIF();
   int initialize(const char * name, HyPerCol * hc, const char * kernel_name);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_Vrest(enum ParamsIOFlag ioFlag);
   virtual void ioParam_Vexc(enum ParamsIOFlag ioFlag);
   virtual void ioParam_Vinh(enum ParamsIOFlag ioFlag);
   virtual void ioParam_VinhB(enum ParamsIOFlag ioFlag);
   virtual void ioParam_VthRest(enum ParamsIOFlag ioFlag);
   virtual void ioParam_tau(enum ParamsIOFlag ioFlag);
   virtual void ioParam_tauE(enum ParamsIOFlag ioFlag);
   virtual void ioParam_tauI(enum ParamsIOFlag ioFlag);
   virtual void ioParam_tauIB(enum ParamsIOFlag ioFlag);
   virtual void ioParam_tauVth(enum ParamsIOFlag ioFlag);
   virtual void ioParam_deltaVth(enum ParamsIOFlag ioFlag);
   virtual void ioParam_deltaGIB(enum ParamsIOFlag ioFlag);
   virtual void ioParam_noiseAmpE(enum ParamsIOFlag ioFlag);
   virtual void ioParam_noiseAmpI(enum ParamsIOFlag ioFlag);
   virtual void ioParam_noiseAmpIB(enum ParamsIOFlag ioFlag);
   virtual void ioParam_noiseFreqE(enum ParamsIOFlag ioFlag);
   virtual void ioParam_noiseFreqI(enum ParamsIOFlag ioFlag);
   virtual void ioParam_noiseFreqIB(enum ParamsIOFlag ioFlag);
   virtual void ioParam_method(enum ParamsIOFlag ioFlag);
   virtual int allocateBuffers();
   virtual int allocateConductances(int num_channels);
   virtual int readStateFromCheckpoint(const char * cpDir, double * timeptr);
   virtual int readVthFromCheckpoint(const char * cpDir, double * timeptr);
   virtual int readG_EFromCheckpoint(const char * cpDir, double * timeptr);
   virtual int readG_IFromCheckpoint(const char * cpDir, double * timeptr);
   virtual int readG_IBFromCheckpoint(const char * cpDir, double * timeptr);
   virtual int readRandStateFromCheckpoint(const char * cpDir, double * timeptr);


private:
   int initialize_base();
   int findPostSynaptic(int dim, int maxSize, int col,
   // input: which layer, which neuron
   HyPerLayer *lSource, float pos[],

   // output: how many of our neurons are connected.
   // an array with their indices.
   // an array with their feature vectors.
   int* nNeurons, int nConnectedNeurons[], float *vPos);
}; // class LIF

BaseObject * createLIF(char const * name, HyPerCol * hc);

} // namespace PV

#endif /* LIF_HPP_ */
