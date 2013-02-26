/*
 * OjaSTDPConn.h
 *
 *  Created on: Sep 27, 2012
 *      Author: dpaiton et slundquist
 */

#ifndef OJASTDPCONN_H_
#define OJASTDPCONN_H_

//#define SPLIT_PRE_POST
#undef SPLIT_PRE_POST

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

   virtual float maxWeight(int axonID);
   virtual int writeTextWeightsExtra(FILE * fd, int k, int axonID);

   virtual int updateState(double time, double dt);
   virtual int updateAmpLTD();
   virtual int updateWeights(int axonID);
   virtual int scaleWeights();
   virtual int outputState(double time, bool last=false);

   virtual int checkpointRead(const char * cpDir, double * timef);
   virtual int checkpointWrite(const char * cpDir);

   //get functions
   float getPostStdpTr(int k)  {return post_stdp_tr->data[k];}
   float getPostOjaTr(int k)   {return post_oja_tr->data[k];}
   float getPostIntTr(int k)   {return post_int_tr->data[k];}
   float getAmpLTD(int k)      {return ampLTD[k];}
   float getPreStdpTr(int kex,int arborID) {return pre_stdp_tr[arborID]->data[kex];}
   float getPreOjaTr(int kex,int arborID)  {return pre_oja_tr[arborID]->data[kex];}

   pvdata_t ** getPostWeightsp(int axonID, int kPost);

   int getNxpPost() {return nxpPost;}
   int getNypPost() {return nypPost;}
   int getNfpPost() {return nfpPost;}

protected:

   int initialize_base();
   int initialize(const char * name, HyPerCol * hc,
                  HyPerLayer * pre, HyPerLayer * post,
                  const char * filename, InitWeights *weightInit);
   virtual int initPlasticityPatches();
#ifdef PV_USE_OPENCL
   virtual int initializeThreadBuffers(const char * kernelName);
   virtual int initializeThreadKernels(const char * kernelName);
#endif // PV_USE_OPENCL

   PVLayerCube * post_stdp_tr; // plasticity decrement variable for postsynaptic layer
   PVLayerCube * post_oja_tr;  // plasticity decrement variable for longer time-constant
   PVLayerCube * post_int_tr;  // plasticity decrement variable for longer time-constant

   //Need pre trace per arbor
   PVLayerCube ** pre_stdp_tr;  // plasticity increment variable for presynaptic layer
   PVLayerCube ** pre_oja_tr;   // plasticity increment variable for presynaptic layer with longer time-constant
   MPI_Datatype * mpi_datatype;  // Used to mirror the presynaptic traces

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
//   float dWMax;
   float targetPostRateHz;
   float targetPreRateHz;
   float weightScale;
   float LTDscale;

   bool  ojaFlag;
   bool  synscalingFlag;
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
