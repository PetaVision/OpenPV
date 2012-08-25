/*
 * STDPConn.hpp
 *
 *  Created on: Jan 28, 2011
 *      Author: sorenrasmussen
 */

#ifndef STDPCONN_HPP_
#define STDPCONN_HPP_

#include "HyPerConn.hpp"
#include "../include/default_params.h"
#include <stdio.h>

namespace PV {

class STDPConn : HyPerConn {
public:
   STDPConn();
   STDPConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
            const char * filename=NULL, bool stdpFlag=true,
            InitWeights *weightInit=NULL);
   virtual ~STDPConn();

   int setParams(PVParams * params);

   virtual int initializeThreadBuffers();
   virtual int initializeThreadKernels();

   // virtual int deleteWeights(); // Changed to a private method.  Should not be virtual since it's called from the destructor.

   virtual float maxWeight(int arborId = 0);

   virtual int updateState(float time, float dt);
   virtual int updateWeights(int axonId);
   virtual int outputState(float time, bool last=false);
   virtual int writeTextWeightsExtra(FILE * fd, int k, int arborID);

   virtual PVLayerCube * getPlasticityDecrement();

protected:

   int initialize_base();
   int initialize(const char * name, HyPerCol * hc,
                  HyPerLayer * pre, HyPerLayer * post,
                  const char * filename, bool stdpFlag, InitWeights *weightInit);
   virtual int initPlasticityPatches();

   PVLayerCube    * post_tr;      // plasticity decrement variable for postsynaptic layer
   PVLayerCube    * pre_tr;      // plasticity increment variable for presynaptic layer
#ifdef OBSOLETE_STDP
   PVPatch       *** dwPatches;      // list of stdp patches Psij variable
#endif

   bool       stdpFlag;         // presence of spike timing dependent plasticity

   int pvpatch_update_plasticity_incr(int nk, float * RESTRICT p,
                                      float aj, float decay, float fac);
   int pvpatch_update_weights(int nk, float * RESTRICT w, const float * RESTRICT m,
                              const float * RESTRICT p, float aPre,
                              const float * RESTRICT aPost, float dWmax, float wMin, float wMax);
#ifdef OBSOLETE
   int pvpatch_update_weights_localWMax(int nk, float * RESTRICT w, const float * RESTRICT m,
                              const float * RESTRICT p, float aPre,
                              const float * RESTRICT aPost, float dWMax, float wMin, float * RESTRICT Wmax);
#endif // OBSOLETE

   // STDP parameters for modifying weights
   float ampLTP; // long term potentiation amplitude
   float ampLTD; // long term depression amplitude
   float tauLTP;
   float tauLTD;
   float dWMax;
   bool synscalingFlag;
   float synscaling_v;

private:
   int deleteWeights();

};

}

#endif /* STDPCONN_HPP_ */
