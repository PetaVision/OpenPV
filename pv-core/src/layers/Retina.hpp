/*
 * Retina.h
 *
 *  Created on: Jul 29, 2008
 *
 */

#ifndef RETINA_HPP_
#define RETINA_HPP_

#include "HyPerLayer.hpp"
#include "../kernels/Retina_params.h"
#include "../io/fileio.hpp"
#include "../arch/opencl/pv_opencl.h"
#include "../include/pv_types.h"

#ifdef PV_USE_OPENCL
#include "../arch/opencl/CLBuffer.hpp"
#endif

#define NUM_RETINA_CHANNELS 2
#define NUM_RETINA_EVENTS   3
//#define EV_R_PHI_E    0
//#define EV_R_PHI_I    1
//#define EV_R_ACTIVITY 2

namespace PV
{

class Retina: public PV::HyPerLayer
{
public:

   friend int test_kernels(int argc, char * argv[]);

   Retina(const char * name, HyPerCol * hc);
   virtual ~Retina();
   virtual int communicateInitInfo();
   virtual int allocateDataStructures();

   int setRetinaParams(PVParams * p);

   virtual int updateState(double time, double dt);
   virtual int outputState(double time, bool last);
   virtual int updateStateOpenCL(double time, double dt);
   virtual int updateBorder(double time, double dt);
   virtual int waitOnPublish(InterColComm* comm);
   virtual int checkpointWrite(const char * cpDir);

   virtual bool activityIsSpiking() { return spikingFlag; }

protected:
   Retina();
   int initialize(const char * name, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_InitVType(enum ParamsIOFlag ioFlag);
   virtual void ioParam_spikingFlag(enum ParamsIOFlag ioFlag);
   virtual void ioParam_foregroundRate(enum ParamsIOFlag ioFlag);
   virtual void ioParam_backgroundRate(enum ParamsIOFlag ioFlag);
   virtual void ioParam_beginStim(enum ParamsIOFlag ioFlag);
   virtual void ioParam_endStim(enum ParamsIOFlag ioFlag);
   virtual void ioParam_burstFreq(enum ParamsIOFlag ioFlag);
   virtual void ioParam_burstDuration(enum ParamsIOFlag ioFlag);
   virtual void ioParam_refractoryPeriod(enum ParamsIOFlag ioFlag);
   virtual void ioParam_absRefractoryPeriod(enum ParamsIOFlag ioFlag);
   virtual int allocateV();
   //int allocateRandStateRestricted(size_t xCount, size_t yCount, size_t fCount, unsigned int seedStart, unsigned int seedStride);
   //int allocateRandStateBorder(int neighbor, size_t xCount, size_t yCount, size_t fCount, unsigned int seedStart, unsigned int seedStride, int indexStart, int indexStride);
   //int allocateRandState(int neighbor, size_t xCount, size_t yCount, size_t fCount, unsigned int seedStart, unsigned int seedStride);
   //int allocateBorderIndices(int neighbor, size_t xCount, size_t yCount, size_t fCount, int indexStart, int indexStride);
   virtual int initializeV();
   virtual int initializeActivity();
//#ifdef PV_USE_OPENCL
//   //int initializeGPU();  //right now there's no use for a Retina specific version
//   virtual int getNumCLEvents() {return numEvents;}
//   virtual const char * getKernelName() {
//      return spikingFlag ? "Retina_spiking_update_state" : "Retina_nonspiking_update_state";
//   }
//   virtual int initializeThreadBuffers(const char * kernel_name);
//   virtual int initializeThreadKernels(const char * kernel_name);
//   //virtual int getEVActivity() {return EV_R_ACTIVITY;}
//
//   CLBuffer * clRand;
//#endif
   virtual int readStateFromCheckpoint(const char * cpDir, double * timeptr);
   virtual int readRandStateFromCheckpoint(const char * cpDir);

   bool spikingFlag;        // specifies that layer is spiking
   Retina_params rParams;   // used in update state
   Random * randState;
//#ifdef PV_USE_OPENCL
//   //TODO-Rasmussen-2014.5.24 - need to figure out interaction between Random class and rand_state
//   taus_uint4 * rand_state[NUM_NEIGHBORHOOD];      // state for random numbers // rand_state[0] for the restricted region; rand_state[1] for northwest corner for background activity, etc.
//#endif
   //size_t rand_state_size[NUM_NEIGHBORHOOD]; // Size of each rand_state pointer.  rand_state_size[0]=numNeurons (local); rand_state_size[NORTHWEST]=nb*nb*nf if the column is in the northwest corner, etc.
   //int * border_indices[NUM_NEIGHBORHOOD];
   float probStimParam;
   float probBaseParam;

private:

   int initialize_base();

   // For input from a given source input layer, determine which
   // cells in this layer will respond to the input activity.
   // Return the feature vectors for both the input and the sensitive
   // neurons, since most likely we will have to determine those.
   int findPostSynaptic(int dim, int maxSize, int col,
      // input: which layer, which neuron
      HyPerLayer *lSource, float pos[],
      // output: how many of our neurons are connected.
      // an array with their indices.
      // an array with their feature vectors.
      int * nNeurons, int nConnectedNeurons[], float * vPos);

   int calculateWeights(HyPerLayer * lSource, float * pos, float * vPos, float * vfWeights );

}; // class Retina

BaseObject * createRetina(char const * name, HyPerCol * hc);

} // namespace PV

#endif /* RETINA_HPP_ */
