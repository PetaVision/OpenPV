/*
 * STDPConn.cpp
 *
 *  Created on: Jan 28, 2011
 *      Author: sorenrasmussen
 */

#include "STDPConn.hpp"
#include "../layers/LIF.hpp"
#include "../io/io.h"
#include <assert.h>

namespace PV {

STDPConn::STDPConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
                   ChannelType channel, const char * filename, bool stdpFlag) : HyPerConn()
{
   initialize_base();
   initialize(name, hc, pre, post, channel, filename, stdpFlag);
}

STDPConn::~STDPConn()
{
   deleteWeights();
}

int STDPConn::initialize_base() {
   // Default STDP parameters for modifying weights; defaults are overridden in setParams().
   this->pIncr = NULL;
   this->pDecr = NULL;
   this->ampLTP = 1.0;
   this->ampLTD = 1.1;
   this->tauLTP = 20;
   this->tauLTD = 20;
   this->dWMax = 0.1;
   this->localWmaxFlag = false;
   return PV_SUCCESS;
}

int STDPConn::initialize(const char * name, HyPerCol * hc,
                         HyPerLayer * pre, HyPerLayer * post,
                         ChannelType channel, const char * filename, bool stdpFlag)
{
   this->stdpFlag = stdpFlag; // needs to be before call to HyPerConn::initialize since it calls overridden methods that depend on stdpFlag being set.
   int status = HyPerConn::initialize(name, hc, pre, post, channel, filename);
   status |= setParams(hc->parameters()); // needs to be called after HyPerConn::initialize since it depends on post being set.

   return status;
}

int STDPConn::initPlasticityPatches() {
   if (stdpFlag) {
      const int arbor = 0;
      pIncr = createWeights(NULL, numWeightPatches(arbor), nxp, nyp, nfp);
      assert(pIncr != NULL);
      pDecr = pvcube_new(&post->getCLayer()->loc, post->getNumExtended());
      assert(pDecr != NULL);
   }

   return PV_SUCCESS;
}

int STDPConn::deleteWeights()
{
   if (stdpFlag) {
      const int arbor = 0;
      const int numPatches = numWeightPatches(arbor);
      for (int k = 0; k < numPatches; k++) {
         pvpatch_inplace_delete(pIncr[k]);
      }
      free(pIncr);
      pvcube_delete(pDecr);
   }
   return 0;
}

int STDPConn::initializeThreadBuffers()
{
   return 0;
}

int STDPConn::initializeThreadKernels()
{
   return 0;
}

PVPatch * STDPConn::getPlasticityPatch(int k, int arbor)
{
   // a separate arbor/patch of plasticity for every neuron
   if (stdpFlag) {
      return pIncr[k];
   }
   return NULL;
}

PVLayerCube * STDPConn::getPlasticityDecrement()
{
   return pDecr;
}

// set member variables specified by user
/*
 * Using a dynamic_cast operator to convert (downcast) a pointer to a base class (HyPerLayer)
 * to a pointer to a derived class (LIF). This way I do not need to define a virtual
 * function getWmax() in HyPerLayer which only returns a NULL pointer in the base class.
 */
int STDPConn::setParams(PVParams * filep)
{
   // stdpFlag is now set by constructor
   // stdpFlag = (bool) filep->value(getName(), "stdpFlag", (float) stdpFlag);

   if (stdpFlag) {
      ampLTP = filep->value(getName(), "ampLTP", ampLTP);
      ampLTD = filep->value(getName(), "ampLTD", ampLTD);
      tauLTP = filep->value(getName(), "tauLTP", tauLTP);
      tauLTD = filep->value(getName(), "tauLTD", tauLTD);

      dWMax = filep->value(getName(), "dWMax", dWMax);

      // set params for rate dependent Wmax
      localWmaxFlag = (bool) filep->value(getName(), "localWmaxFlag", (float) localWmaxFlag);
   }
   if (localWmaxFlag){ // Not sure if the if(localWmaxFlag) should be inside or outside the if(stdpFlag) statement
      LIF * LIF_layer = dynamic_cast<LIF *>(post);
      assert(LIF_layer != NULL);
      Wmax = LIF_layer->getWmax();
      assert(Wmax != NULL);
   } else {
      Wmax = NULL;
   }

   return 0;
}

int STDPConn::updateState(float time, float dt)
{
   update_timer->start();

   if (stdpFlag) {
      const float fac = ampLTD;
      const float decay = expf(-dt / tauLTD);

      //
      // both pDecr and activity are extended regions (plus margins)
      // to make processing them together simpler

      const int nk = pDecr->numItems;
      const float * a = post->getLayerData();
      float * m = pDecr->data; // decrement (minus) variable

      for (int k = 0; k < nk; k++) {
         m[k] = decay * m[k] - fac * a[k];
      }

      const int axonId = 0;       // assume only one for now
      updateWeights(axonId);
   }
   update_timer->stop();

   const int axonId = 0;       // assume only one for now
   return updateWeights(axonId);
}

/**
 *  M (m or pDecr->data) is an extended post-layer variable
 *
 */
int STDPConn::updateWeights(int axonId)
{
   // Steps:
   // 1. Update pDecr (assume already done as it should only be done once)
   // 2. update Psij (pIncr) for each synapse
   // 3. update wij

   const float dt = parent->getDeltaTime();
   const float decayLTP = expf(-dt / tauLTP);

   const int numExtended = pre->getNumExtended();
   assert(numExtended == numWeightPatches(axonId));

   const pvdata_t * preLayerData = pre->getLayerData();

   // this stride is in extended space for post-synaptic activity and
   // STDP decrement variable
   const int postStrideY = post->getLayerLoc()->nf
                         * (post->getLayerLoc()->nx + 2 * post->getLayerLoc()->nb);

   for (int kPre = 0; kPre < numExtended; kPre++) {
      PVAxonalArbor * arbor = axonalArbor(kPre, axonId);

      const float preActivity = preLayerData[kPre];

      PVPatch * pIncr   = arbor->plasticIncr;
      PVPatch * w       = arbor->weights;
      size_t postOffset = arbor->offset;

      const float * postActivity = &post->getLayerData()[postOffset];
      const float * M = &pDecr->data[postOffset];  // STDP decrement variable
      float * P = pIncr->data;                     // STDP increment variable
      float * W = w->data;

      int nk  = pIncr->nf * pIncr->nx; // one line in x at a time
      int ny  = pIncr->ny;
      int sy  = pIncr->sy;

      // TODO - unroll

      // update Psij (pIncr variable)
      // we are processing patches, one line in y at a time
      for (int y = 0; y < ny; y++) {
         pvpatch_update_plasticity_incr(nk, P + y * sy, preActivity, decayLTP, ampLTP);
      }

      if (localWmaxFlag) {
         // update weights with local post-synaptic Wmax values
         // Wmax lives in the restricted space - it is controlled
         // by average rate in the post synaptic layer
         float * Wmax_pointer = &Wmax[postOffset];
         for (int y = 0; y < ny; y++) {
            // TODO
            pvpatch_update_weights_localWMax(nk,W,M,P,preActivity,postActivity,dWMax,wMin,Wmax_pointer);
            //
            // advance pointers in y
            W += sy;
            P += sy;

            //
            // postActivity, M are extended post-layer variables
            //
            postActivity += postStrideY;
            M            += postStrideY;
            Wmax_pointer += postStrideY;
         }
      } else {
         // update weights
         for (int y = 0; y < ny; y++) {
            pvpatch_update_weights(nk, W, M, P, preActivity, postActivity, dWMax, wMin, wMax);
            //
            // advance pointers in y
            W += sy;
            P += sy;
            //
            // postActivity and M are extended layer
            postActivity += postStrideY;
            M += postStrideY;
         }

      }
   }

   return 0;
}

int STDPConn::outputState(float time, bool last)
{
   int status = HyPerConn::outputState(time, last);
   if (status != PV_SUCCESS) return status;

   if (stdpFlag != true) return status;

   if (last) {
      convertPreSynapticWeights(time);
      status = writePostSynapticWeights(time, last);
      assert(status == 0);
   }
   else if ( (time >= writeTime) && (writeStep >= 0) ) {
      convertPreSynapticWeights(time);
      status = writePostSynapticWeights(time, last);
      assert(status == 0);
   }

   return status;
}

float STDPConn::maxWeight()
{
   float maxVal = 0.0;

   if (localWmaxFlag) {
      const int numExtended = post->getNumExtended();
      for (int kPost = 0; kPost < numExtended; kPost++){
         if(Wmax[kPost] > maxVal){
            maxVal = Wmax[kPost];
         }
      }
   } else {
      maxVal = wMax;
   }

   return maxVal;
}

int STDPConn::writeTextWeightsExtra(FILE * fd, int k)
{
   if (stdpFlag) {
      pv_text_write_patch(fd, pIncr[k]); // write the Ps variable
   }
   return 0;
}

int STDPConn::adjustAxonalPatches(PVAxonalArbor * arbor, int nxPatch, int nyPatch, int dx, int dy)
{
   int status = HyPerConn::adjustAxonalPatches(arbor, nxPatch, nyPatch, dx, dy);

   if (stdpFlag && status == PV_SUCCESS) {
      pvpatch_adjust(arbor->plasticIncr, nxPatch, nyPatch, dx, dy);
   }

   return status;
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
 * Loop over presynaptic extended layer.  Calculate pIncr, and weights.
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
#endif // NOTYET - TODO


} // End of namespace PV

