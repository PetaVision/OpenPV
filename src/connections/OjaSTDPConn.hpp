/*
 * OjaSTDPConn.h
 *
 *  Created on: Sep 27, 2012
 *      Author: dpaiton
 */

#ifndef OJASTDPCONN_H_
#define OJASTDPCONN_H_

#include "HyPerConn.hpp"
#include "../include/default_params.h"
#include <stdio.h>

namespace PV {

class OjaSTDPConn: HyPerConn {
public:
   OjaSTDPConn();
   OjaSTDPConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
            const char * filename=NULL, bool stdpFlag=true,
            InitWeights *weightInit=NULL);
   virtual ~OjaSTDPConn();

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
   PVLayerCube    * post_sum_tr;      // summed spikes for reconstruction term
   PVLayerCube    * pre_tr;      // plasticity increment variable for presynaptic layer
   PVLayerCube    * pre_sum_tr;      // summed spikes (longer time constant) for reconstruction term

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

   pvdata_t * prevW;
   // STDP parameters for modifying weights
   float ampLTP; // long term potentiation amplitude
   float ampLTD; // long term depression amplitude
   float tauLTP;
   float tauLTD;
   float tauLTPLong;
   float tauLTDLong;
   float weightDecay;
   float dWMax;
   bool  synscalingFlag;
   float synscaling_v;

private:
   int deleteWeights();

};

}

#endif /* OJASTDPCONN_H_ */
