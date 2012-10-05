/*
 * OjaOjaSTDPConn.cpp
 *
 *  Created on: Sep 27, 2012
 *      Author: dpaiton et slundquist
 */

#include "OjaSTDPConn.hpp"
#include "../layers/LIF.hpp"
#include "../io/io.h"
#include <assert.h>
#include <math.h>

namespace PV {

OjaSTDPConn::OjaSTDPConn(const char * name, HyPerCol * hc, HyPerLayer * pre,
      HyPerLayer * post, const char * filename, bool stdpFlag, InitWeights *weightInit)
{
   initialize_base();
   initialize(name, hc, pre, post, filename, stdpFlag, weightInit);
#ifdef PV_USE_OPENCL
   if(gpuAccelerateFlag)
      initializeGPU();
#endif
}

OjaSTDPConn::~OjaSTDPConn()
{
   deleteWeights();
}

int OjaSTDPConn::initialize_base() {
   // Default STDP parameters for modifying weights; defaults are overridden in setParams().
   // this->dwPatches = NULL;
   this->post_tr      = NULL;
   this->ampLTP       = 0.0065; //amp sets ratio of LTP to LTD, or how much more/less effective LTP is than LTD. LTP/LTD should ~= 0.9 per Gar
   this->ampLTD       = 0.0071;
   this->tauLTP       = 16.8;
   this->tauLTD       = 33.7;
   this->tauLTPLong   = 168;
   this->tauLTDLong   = 33.7;
   this->weightDecay  = 0.01;
   this->dWMax        = 1;
   this->ojaScale     = 1;
   this->STDPScale    = 1;
   this->wMax         = 0.0001;

   this->synscalingFlag = false;
   this->synscaling_v   = 1;

   return PV_SUCCESS;
}

int OjaSTDPConn::initialize(const char * name, HyPerCol * hc,
      HyPerLayer * pre, HyPerLayer * post,
      const char * filename, bool stdpFlag, InitWeights *weightInit)
{
   this->stdpFlag = stdpFlag; //needs to be before call to HyPerConn::initialize since it calls overridden methods that depend on stdpFlag being set.
   int status = HyPerConn::initialize(name, hc, pre, post, filename, weightInit);

   status |= setParams(hc->parameters()); // needs to be called after HyPerConn::initialize since it depends on post being set
   status |= initPlasticityPatches();

   if(synscalingFlag){
      point2PreSynapticWeights(0);
   }

   return status;
}

int OjaSTDPConn::initPlasticityPatches()
{
   if (!stdpFlag) return PV_SUCCESS;

   int status = HyPerConn::initPlasticityPatches();
   assert(status == 0);

   post_tr      = pvcube_new(&post->getCLayer()->loc, post->getNumExtended());
   pre_tr       = pvcube_new(&pre->getCLayer()->loc, pre->getNumExtended());
   assert(post_tr      != NULL);
   assert(pre_tr       != NULL);

   return PV_SUCCESS;
}

int OjaSTDPConn::deleteWeights()
{
   if (stdpFlag) {
      pvcube_delete(post_tr);
      post_tr = NULL;
   }
   return 0;
}

int OjaSTDPConn::initializeThreadBuffers() {return 0;}
int OjaSTDPConn::initializeThreadKernels() {return 0;}

// set member variables specified by user
/*
 * Using a dynamic_cast operator to convert (downcast) a pointer to a base class (HyPerLayer)
 * to a pointer to a derived class (LIF). This way I do not need to define a virtual
 * function getWmax() in HyPerLayer which only returns a NULL pointer in the base class.
 */
int OjaSTDPConn::setParams(PVParams * params)
{
   // stdpFlag is now set by constructor
   HyPerConn::setParams(params);
   if (stdpFlag) {
      ampLTP         = params->value(getName(), "ampLTP", ampLTP);
      ampLTD         = params->value(getName(), "ampLTD", ampLTD);
      tauLTP         = params->value(getName(), "tauLTP", tauLTP);
      tauLTD         = params->value(getName(), "tauLTD", tauLTD);
      tauLTDLong     = params->value(getName(), "tauLTDLong", tauLTDLong);
      tauLTPLong     = params->value(getName(), "tauLTPLong", tauLTPLong);
      weightDecay    = params->value(getName(), "weightDecay", weightDecay);
      ojaScale       = params->value(getName(), "ojaScale", ojaScale);
      STDPScale      = params->value(getName(), "STDPScale", ojaScale);

      wMax           = params->value(getName(), "wMax", wMax);
      wMin           = params->value(getName(), "wMin", wMin);
      dWMax          = params->value(getName(), "dWMax", dWMax);
      synscalingFlag = params->value(getName(), "synscalingFlag", synscalingFlag);
      synscaling_v   = params->value(getName(), "synscaling_v", synscaling_v);
   }
   return 0;
}

/**
 * First function to be executed
 * Updates the postsynaptic trace and calls the updateWeights function
 */
int OjaSTDPConn::updateState(float time, float dt)
{
   update_timer->start();

   int status=0;
   if (stdpFlag) {
      for(int axonID = 0; axonID<numberOfAxonalArborLists(); axonID++) {
         status=updateWeights(axonID);
      }
   }

   update_timer->stop();

   return status;
}

int OjaSTDPConn::updateWeights(int axonID)
{
   // Steps:
   // 1. Update post_tr
   // 2. Update pre_tr
   // 3. Update w_ij

   const float dt           = parent->getDeltaTime();
   const float decayLTP     = exp(-dt / tauLTP);
   const float decayLTD     = exp(-dt / tauLTD);
   const float decayLTDLong = exp(-dt / tauLTDLong);
   const float decayLTPLong = exp(-dt / tauLTPLong);

   const int nkPost = post_tr->numItems;
   const int nkpre  = pre->getNumExtended();
   assert(nkpre == getNumWeightPatches());

   const pvdata_t * preLayerData = pre->getLayerData(getDelay(axonID));
   const pvdata_t * aPost        = post->getLayerData();
   pvdata_t aPre;

   pvdata_t * post_tr_m;      // Postsynaptic trace matrix; i.e. data of post_tr struct
   pvdata_t * post_long_tr_m; // Postsynaptic mean trace matrix
   pvdata_t * pre_tr_m;       // Presynaptic trace matrix
   pvdata_t * pre_long_tr_m;
   pvdata_t * W;              // Weight matrix pointer

   int nk, ny;

   post_tr_m      = post_tr->data;
   post_long_tr_m = post_tr->data;

   // 1. Updates the postsynaptic traces
   for (int kPost = 0; kPost < nkPost; kPost++)
   {
      post_tr_m[kPost]      = decayLTD * post_tr_m[kPost] + aPost[kPost];
      post_long_tr_m[kPost] = decayLTDLong * post_long_tr_m[kPost] + aPost[kPost];
   }

   // this stride is in extended space for post-synaptic activity and STDP decrement variable
   const int postStrideY = post->getLayerLoc()->nf * (post->getLayerLoc()->nx + 2 * post->getLayerLoc()->nb);
   //FIXME: In the first iteration post is -70!! (May not still be true)

   for (int kPre = 0; kPre < nkpre; kPre++)              // Loop over all presynaptic neurons
   {
      size_t postOffset = getAPostOffset(kPre, axonID);  // Gets start index for postsynaptic vectors for given presynaptic neuron and axon

      aPre           = preLayerData[kPre];                // Spiking activity
      aPost          = &post->getLayerData()[postOffset]; // Gets address of postsynaptic activity
      post_tr_m      = &(post_tr->data[postOffset]);      // Reference to STDP post trace
      post_long_tr_m = &(post_tr->data[postOffset]);
      pre_tr_m       = &(pre_tr->data[kPre]);             // PreTrace for given presynaptic neuron kPre
      pre_long_tr_m  = &(pre_tr->data[kPre]);

      W = get_wData(axonID, kPre);                       // Pointer to data of given axon & presynaptic neuron

      // Get weights in form of a patch (nx,ny,nf)
      // nk and ny are the number of neurons connected to the given presynaptic neuron in the x*nfp and y
      // if each of the presynaptic neurons connects to all postsynaptic than nk*ny = nkPost TODO: Is this true? Rui says yes.
      PVPatch * w = getWeights(kPre, axonID);                // Get weights in form of a patch (nx,ny,nf), TODO: what's the role of the offset?
      nk  = nfp * w->nx; // one line in x at a time
      ny  = w->ny;

      // 2. Updates the presynaptic trace
      *pre_tr_m      = decayLTP * (*pre_tr_m) + aPre;        //If spiked, minimum is 1. If no spike, minimum is 0.
      *pre_long_tr_m = decayLTPLong * (*pre_long_tr_m) + aPre;

      //3. Update weights
      for (int y = 0; y < ny; y++) {
         for (int k = 0; k < nk; k++) {
            //deltaQmnt ~ [Xm'(t) - Yn'(t) * Qmnt-1] * [a * Ay(t) * Xm(t) - Ax(t) * Yn(t)] - l*Qmnt-1
            // Xm(t), Yn(t) = pre & post Oja trace (respectively)
            // Xm'(t), Yn'(t) = pre & post trace, but on a longer time scale (to get a more integrated trace)
            // Qmnt is weight at current time step
            // Qmnt-1 is weight at previous time step
            // Ax(t),Ay(t) is spike activity for pre/post respectively
            W[k] += dWMax * (((*pre_long_tr_m) - post_long_tr_m[k] * W[k]) *
                  (ampLTP * aPost[k] * (*pre_tr_m) - ampLTD * aPre * post_tr_m[k]) - weightDecay * W[k]);

            W[k] = W[k] < wMin ? wMin : W[k]; // Stop weights from going all the way to 0
            W[k] = W[k] > wMax ? wMax : W[k]; //FIXME: No need for a max now that we have the decay terms and oja rule??
         }

         // advance pointers in y
         W += syp; //FIXME: W += nk

         // postActivity and post trace are extended layer
         aPost          += postStrideY; //TODO: is this really in the extended space?
         post_tr_m      += postStrideY;
         post_long_tr_m += postStrideY;
      }
   }

   if(synscalingFlag){
      //int kxPre, kyPre, kPre;

      const int numPostPatch = nxpPost * nypPost * nfpPost; // Post-synaptic weights are never shrunken

      float sumW = 0;
      //int kxPost, kyPost, kfPost;
      const int xScale = post->getXScale() - pre->getXScale();
      const int yScale = post->getYScale() - pre->getYScale();
      const double powXScale = pow(2.0f, (double) xScale);
      const double powYScale = pow(2.0f, (double) yScale);

      nxpPost = (int) (nxp * powXScale);
      nypPost = (int) (nyp * powYScale);
      nfpPost = pre->clayer->loc.nf;

      for(int axonID=0;axonID<numberOfAxonalArborLists();axonID++) {
         //Loop through post-synaptic neurons (non-extended indices)
         for (int kPost = 0; kPost < post_tr->numItems; kPost++) {

            pvdata_t ** postData = wPostDataStartp[axonID] + numPostPatch*kPost + 0;
            for (int kp = 0; kp < numPostPatch; kp++) { //TODO: Scale only the weights non-extended space
               sumW += *(postData[kp]);
            }
            for (int kp = 0; kp < numPostPatch; kp++) {
               *(postData[kp]) = ((*postData[kp])/sumW)*synscaling_v;
            }
            //printf("%f ",sumW);
            sumW = 0;
         }
         //printf("\n");
      }
   }

   return 0;
}

int OjaSTDPConn::outputState(float timef, bool last)
{
   int status;

   if (last) {
      printf("Writing last STDP weights..%f\n",timef);
      convertPreSynapticWeights(timef);
      status = writePostSynapticWeights(timef, last);
      assert(status == 0);
   }else if ( (timef >= writeTime) && (writeStep >= 0) ) {
      //writeTime += writeStep; Done in HyperConn
      convertPreSynapticWeights(timef);
      status = writePostSynapticWeights(timef, false);
      assert(status == 0);

      // append to output file after original open
      //ioAppend = true;
   }
   status = HyPerConn::outputState(timef, last);

   if (status != PV_SUCCESS) return status;


   //   if (stdpFlag != true) return status;
   //
   //   if (last) {
   //      convertPreSynapticWeights(time);
   //      status = writePostSynapticWeights(time, last);
   //      assert(status == PV_SUCCESS);
   //   }
   //   else if ( (time >= writeTime) && (writeStep >= 0) ) {
   //
   //      convertPreSynapticWeights(time);
   //      status = writePostSynapticWeights(time, last);
   //      assert(status == PV_SUCCESS);
   //   }

   return status;
}

float OjaSTDPConn::maxWeight(int axonID)
{
   return wMax;
}

int OjaSTDPConn::writeTextWeightsExtra(FILE * fd, int k, int axonID)
{
   if (stdpFlag) {
      pv_text_write_patch(fd, getWeights(k, axonID), get_dwData(axonID, k), nfp, sxp, syp, sfp); // write the Ps variable
   }
   return 0;
}

#ifdef NOTYET
void STDP_update_state_post(
      const float dt,

      const int nx,
      const int ny,
      const int nf,
      const int nb,

      const int nxp,
      const int nyp,

      STDP_params * params,

      float * M,
      float * Wmax,
      float * Apost,
      float * Rpost)
{

   int kex;
#ifndef PV_USE_OPENCL
   for (kex = 0; kex < nx*ny*nf; kex++) {
#else
      kex = get_global_id(0);
#endif

      //
      // kernel (nonheader part) begins here
      //

      // update the decrement variable
      //
      M[kex] = decay * M[kex] - fac * Apost[kex];

#ifndef PV_USE_OPENCL
   }
#endif

}


/**
 * Loop over presynaptic extended layer.  Calculate dwPatches, and weights.
 */
void STDP_update_state_pre(
      const float time,
      const float dt,

      const int nx,
      const int ny,
      const int nf,
      const int nb,

      const int nxp,
      const int nyp,

      STDP_params * params,

      float * M,
      float * P,
      float * W,
      float * Wmax,
      float * Apre,
      float * Apost)
{

   int kex;

   float m[NXP*NYP], aPost[NXP*NYP], wMax[NXP*NYP];

#ifndef PV_USE_OPENCL
   for (kex = 0; kex < nx*ny*nf; kex++) {
#else
      kex = get_global_id(0);
#endif

      //
      // kernel (nonheader part) begins here
      //

      // update the increment variable
      //
      float aPre = Apre[kex];
      float * p = P[kex*stride];

      // copy into local variable
      //

      copy(m, M);
      copy(aPost, Apost);
      copy(wMax, Wmax);

      // update the weights
      //
      for (int kp = 0; kp < nxp*nyp; kp++) {
         p[kp] = decay * p[kp] + ltpAmp * aPre;
         w[kp] += dWMax * (aPre * m[kp] + aPost[kp] * p[kp]);
         w[kp] = w[kp] < wMin ? wMin : w[kp];
         w[kp] = w[kp] > wMax ? wMax : w[kp];
      }
#ifndef PV_USE_OPENCL
   }
#endif

}
#endif //NOTYET - TODO

int OjaSTDPConn::checkpointRead(const char * cpDir, float* timef) {
   int status = HyPerConn::checkpointRead(cpDir, timef);
   char filename[PV_PATH_MAX];
   int chars_needed;
   double timed;
   PVLayerLoc loc;

   memcpy(&loc, pre->getLayerLoc(), sizeof(PVLayerLoc));
   loc.nx += 2*loc.nb;
   loc.ny += 2*loc.nb;
   loc.nxGlobal = loc.nx * parent->icCommunicator()->numCommColumns();
   loc.nyGlobal = loc.ny * parent->icCommunicator()->numCommRows();
   loc.nb = 0;
   chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_pre_tr.pvp", cpDir, name);
   if (chars_needed >= PV_PATH_MAX) {
      fprintf(stderr, "LCALIFLateralConn::checkpointWrite error.  Path \"%s/%s_pre_tr.pvp\" is too long.\n", cpDir, name);
      abort();
   }
   read_pvdata(filename, parent->icCommunicator(), &timed, pre_tr->data, &loc, PV_FLOAT_TYPE, /*extended*/ false, /*contiguous*/ false);

   memcpy(&loc, post->getLayerLoc(), sizeof(PVLayerLoc));
   loc.nx += 2*loc.nb;
   loc.ny += 2*loc.nb;
   loc.nxGlobal = loc.nx * parent->icCommunicator()->numCommColumns();
   loc.nyGlobal = loc.ny * parent->icCommunicator()->numCommRows();
   loc.nb = 0;
   chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_post_tr.pvp", cpDir, name);
   if (chars_needed >= PV_PATH_MAX) {
      fprintf(stderr, "LCALIFLateralConn::checkpointWrite error.  Path \"%s/%s_post_tr.pvp\" is too long.\n", cpDir, name);
      abort();
   }
   read_pvdata(filename, parent->icCommunicator(), &timed, post_tr->data, &loc, PV_FLOAT_TYPE, /*extended*/ false, /*contiguous*/ false);

   if( (float) timed != *timef && parent->icCommunicator()->commRank() == 0 ) {
      fprintf(stderr, "Warning: %s and %s_A.pvp have different timestamps: %f versus %f\n", filename, name, (float) timed, *timef);
   }

   return status;
}

int OjaSTDPConn::checkpointWrite(const char * cpDir) {
   int status = HyPerConn::checkpointWrite(cpDir);
   // This is kind of hacky, but we save the extended buffers post_tr, pre_tr as if they were nonextended buffers of size (nx+2*nb)-by-(ny+2*nb)
   char filename[PV_PATH_MAX];
   int chars_needed;
   PVLayerLoc loc;

   memcpy(&loc, pre->getLayerLoc(), sizeof(PVLayerLoc));
   loc.nx += 2*loc.nb;
   loc.ny += 2*loc.nb;
   loc.nxGlobal = loc.nx * parent->icCommunicator()->numCommColumns();
   loc.nyGlobal = loc.ny * parent->icCommunicator()->numCommRows();
   loc.nb = 0;
   chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_pre_tr.pvp", cpDir, name);
   if (chars_needed >= PV_PATH_MAX) {
      fprintf(stderr, "LCALIFLateralConn::checkpointWrite error.  Path \"%s/%s_pre_tr.pvp\" is too long.\n", cpDir, name);
      abort();
   }
   write_pvdata(filename, parent->icCommunicator(), (double) parent->simulationTime(), pre_tr->data, &loc, PV_FLOAT_TYPE, /*extended*/ false, /*contiguous*/ false);

   memcpy(&loc, post->getLayerLoc(), sizeof(PVLayerLoc));
   loc.nx += 2*loc.nb;
   loc.ny += 2*loc.nb;
   loc.nxGlobal = loc.nx * parent->icCommunicator()->numCommColumns();
   loc.nyGlobal = loc.ny * parent->icCommunicator()->numCommRows();
   loc.nb = 0;
   chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_post_tr.pvp", cpDir, name);
   if (chars_needed >= PV_PATH_MAX) {
      fprintf(stderr, "LCALIFLateralConn::checkpointWrite error.  Path \"%s/%s_post_tr.pvp\" is too long.\n", cpDir, name);
      abort();
   }
   write_pvdata(filename, parent->icCommunicator(), (double) parent->simulationTime(), post_tr->data, &loc, PV_FLOAT_TYPE, /*extended*/ false, /*contiguous*/ false);
   return status;
}


} // End of namespace PV
