/*
 * STDPConn.hpp
 *
 *  Created on: Jan 28, 2011
 *      Author: sorenrasmussen
 */

#ifndef STDPCONN_HPP_
#define STDPCONN_HPP_

#include "HyPerConn.hpp"
#include <stdio.h>

namespace PV {

class STDPConn : HyPerConn {
public:
   STDPConn();
   STDPConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
            ChannelType channel, const char * filename=NULL, bool stdpFlag=true);
   virtual ~STDPConn();

   int setParams(PVParams * params);

   virtual int initializeThreadBuffers();
   virtual int initializeThreadKernels();

   virtual int deleteWeights();

   virtual float maxWeight();

   virtual int updateState(float time, float dt);
   virtual int updateWeights(int axonId);
   virtual int outputState(float time, bool last=false);
   virtual int writeTextWeightsExtra(FILE * fd, int k);

   virtual PVPatch     * getPlasticityPatch(int k, int arbor);
   virtual PVLayerCube * getPlasticityDecrement();

protected:

   int initialize_base();
   int initialize(const char * name, HyPerCol * hc,
                  HyPerLayer * pre, HyPerLayer * post,
                  ChannelType channel, const char * filename, bool stdpFlag);
   virtual int initPlasticityPatches();

   virtual int adjustAxonalPatches(PVAxonalArbor * arbor, int nxPatch, int nyPatch, int dx, int dy);

   PVLayerCube    * pDecr;      // plasticity decrement variable (Mi) for post-synaptic layer
   PVPatch       ** pIncr;      // list of stdp patches Psij variable

   bool       stdpFlag;         // presence of spike timing dependent plasticity
   bool       localWmaxFlag;    // presence of rate dependent wMax;
   pvdata_t * Wmax;             // adaptive upper STDP weight boundary

   // STDP parameters for modifying weights
   float ampLTP; // long term potentiation amplitude
   float ampLTD; // long term depression amplitude
   float tauLTP;
   float tauLTD;
   float dWMax;
};

}

#endif /* STDPCONN_HPP_ */
