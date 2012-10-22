/*
 * OjaSTDPConn.h
 *
 *  Created on: Sep 27, 2012
 *      Author: dpaiton et slundquist
 */

#ifndef OJASTDPCONN_H_
#define OJASTDPCONN_H_

#include "HyPerConn.hpp"
#include "../include/default_params.h"
#include <stdio.h>
#include "../io/fileio.hpp"

namespace PV {

class OjaSTDPConn: public HyPerConn {
public:
   OjaSTDPConn();
   OjaSTDPConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
            const char * filename=NULL, InitWeights *weightInit=NULL);
   virtual ~OjaSTDPConn();

   int setParams(PVParams * params);

   virtual int initializeThreadBuffers();
   virtual int initializeThreadKernels();

   virtual float maxWeight(int axonID);
   virtual int writeTextWeightsExtra(FILE * fd, int k, int axonID);

   virtual int updateState(float time, float dt);
   virtual int updateWeights(int axonID);
   virtual int outputState(float time, bool last=false);

   virtual int checkpointRead(const char * cpDir, float* timef);
   virtual int checkpointWrite(const char * cpDir);

   //get functions
   float getPostStdpTr(int k)  {return post_stdp_tr->data[k];}
   float getPostOjaTr(int k)   {return post_oja_tr->data[k];}
   float getPostIntTr(int k)   {return post_int_tr->data[k];}
   float getPreStdpTr(int kex) {return pre_stdp_tr->data[kex];}
   float getPreOjaTr(int kex)  {return pre_oja_tr->data[kex];}
   float getAmpLTD(int k)      {return ampLTD[k];}

   pvdata_t ** getPostWeights(int axonID, int kPost);

   int getNxpPost() {return nxpPost;}
   int getNypPost() {return nypPost;}
   int getNfpPost() {return nfpPost;}

protected:

   int initialize_base();
   int initialize(const char * name, HyPerCol * hc,
                  HyPerLayer * pre, HyPerLayer * post,
                  const char * filename, InitWeights *weightInit);
   virtual int initPlasticityPatches();

   PVLayerCube * post_stdp_tr; // plasticity decrement variable for postsynaptic layer
   PVLayerCube * post_oja_tr;  // plasticity decrement variable for longer time-constant
   PVLayerCube * post_int_tr;  // plasticity decrement variable for longer time-constant
   PVLayerCube * pre_stdp_tr;  // plasticity increment variable for presynaptic layer
   PVLayerCube * pre_oja_tr;   // plasticity increment variable for presynaptic layer with longer time-constant

   float * ampLTD;

   // STDP parameters for modifying weights
   float ampLTP; // long term potentiation amplitude
   float initAmpLTD; // long term depression amplitude
   float tauLTP;
   float tauLTD;
   float tauOja;
   float tauTHR;
   float tauO;
   float weightDecay;
   float dWMax;
   float targetRateHz;
   float LTDscale;

   bool  synscalingFlag;
   bool  ojaFlag;
   float synscaling_v;

#ifdef OBSOLETE_STDP
   PVPatch       *** dwPatches;      // list of stdp patches Psij variable
#endif
#ifdef OBSOLETE
   int pvpatch_update_weights_localWMax(int nk, float * RESTRICT w, const float * RESTRICT m,
                              const float * RESTRICT p, float aPre,
                              const float * RESTRICT aPost, float dWMax, float wMin, float * RESTRICT Wmax);
#endif // OBSOLETE

private:
   int deleteWeights();

};

}

#endif /* OJASTDPCONN_H_ */
