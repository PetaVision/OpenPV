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
#include "../arch/opencl/pv_opencl.h"
#include "../arch/opencl/pv_uint4.h"
#include "Image.hpp"

#define NUM_RETINA_EVENTS 3
#define EV_PHI_E    0
#define EV_PHI_I    1
#define EV_ACTIVITY 2

namespace PV
{

class Retina: public PV::HyPerLayer
{
public:
   Retina(const char * name, HyPerCol * hc);
   virtual ~Retina();

   int initialize(PVLayerType type);
   int setParams(PVParams * p);

   virtual int triggerReceive(InterColComm* comm);
   virtual int updateState (float time, float dt);
   virtual int updateBorder(float time, float dt);
   virtual int waitOnPublish(InterColComm* comm);
#ifdef PV_USE_OPENCL
   virtual int updateStateOpenCL(float time, float dt);
#endif

   virtual int writeState(const char * path, float time);
#ifdef OBSOLETE
   virtual int spike(float time, float dt, float prev, float prob, float probStim, float * probSpike);
#endif

protected:

#ifdef PV_USE_OPENCL
   virtual int initializeThreadBuffers();
   virtual int initializeThreadKernels();
#endif

   bool spikingFlag;        // specifies that layer is spiking
   Retina_params rParams;   // used in update state
   uint4 * rand_state;      // state for random numbers

private:

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

};

} // namespace PV

#endif /* RETINA_HPP_ */
