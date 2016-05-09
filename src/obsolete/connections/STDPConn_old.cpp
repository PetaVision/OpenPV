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
                   ChannelType channel, const char * filename, bool stdpFlag,
                   InitWeights *weightInit) : HyPerConn()
{
   initialize_base();
   initialize(name, hc, pre, post, channel, filename, stdpFlag, weightInit);
}

STDPConn::~STDPConn()
{
   deleteWeights();
}

int STDPConn::initialize_base() {
   // Default STDP parameters for modifying weights; defaults are overridden in setParams().
   // this->dwPatches = NULL;
   this->pDecr = NULL;
   this->ampLTP = 1.0;
   this->ampLTD = 1.1;
   this->tauLTP = 20;
   this->tauLTD = 20;
   this->dWMax = 0.1;
   return PV_SUCCESS;
}

int STDPConn::initialize(const char * name, HyPerCol * hc,
                         HyPerLayer * pre, HyPerLayer * post,
                         ChannelType channel, const char * filename, bool stdpFlag, InitWeights *weightInit)
{
   this->stdpFlag = stdpFlag; // needs to be before call to HyPerConn::initialize since it calls overridden methods that depend on stdpFlag being set.
   int status = HyPerConn::initialize(name, hc, pre, post, channel, filename, weightInit);

   //status |= setParams(hc->parameters()); // needs to be called after HyPerConn::initialize since it depends on post being set
   status |= initPlasticityPatches();

   return status;
}

int STDPConn::initPlasticityPatches()
{
   if (!stdpFlag) return PV_SUCCESS;

   int status = HyPerConn::initPlasticityPatches();
   assert(status == 0);
   //const int arbor = 0;
   //const int numAxons = numberOfAxonalArborLists();

//   dwPatches = createWeights(NULL, numWeightPatches(), nxp, nyp, nfp, 0);
   pDecr = pvcube_new(&post->getCLayer()->loc, post->getNumExtended());
   assert(pDecr != NULL);

   // moved to HyPerConn
#ifdef OBSOLETE_STDP
   dwPatches = (PVPatch***) calloc(numAxons, sizeof(PVPatch**));
   assert(dwPatches != NULL);
   int numArbors = getNumWeightPatches();
   for (int n = 0; n < numAxons; n++) {

      dwPatches[n] = createWeights(NULL, getNumWeightPatches(), nxp, nyp, nfp, 0);
      assert(dwPatches[n] != NULL);


      // kex is in extended frame
      for (int kex = 0; kex < numArbors; kex++) {
         int kl, offset, nxPatch, nyPatch, dx, dy;
         // PVAxonalArbor * arbor = axonalArbor(kex, n);

         calcPatchSize(n, kex, &kl, &offset, &nxPatch, &nyPatch, &dx, &dy);

         // adjust patch size (shrink) to fit within interior of post-synaptic layer
         //
         // arbor->plasticIncr = dwPatches[n][kex];
         pvpatch_adjust(dwPatches[n][kex], nxPatch, nyPatch, dx, dy);

      } // loop over arbors (pre-synaptic neurons)
   } // loop over neighbors
#endif
   return PV_SUCCESS;
}

int STDPConn::deleteWeights()
{
   if (stdpFlag) {
      //dwPatches belongs to HyPerConn, so it is deleted in HyPerConn::deleteWeights()
      // const int numPatches = numWeightPatches();
      // const int numAxons = numberOfAxonalArborLists();
      // for (int n = 0; n < numAxons; n++) {
      //    for (int k = 0; k < numPatches; k++) {
      //       pvpatch_inplace_delete(dwPatches[n][k]);
      //    }
      //    free(dwPatches[n]);
      // }
      // free(dwPatches);
      pvcube_delete(pDecr);
      pDecr = NULL;
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
int STDPConn::ioParams(enum ParamsIOFlag ioFlag)
{
   // stdpFlag is now set by constructor
   // stdpFlag = (bool) filep->value(getName(), "stdpFlag", (float) stdpFlag);
   HyPerConn::ioParams(ioFlag);

   if (stdpFlag) {
      ampLTP = params->value(getName(), "ampLTP", ampLTP);
      ampLTD = params->value(getName(), "ampLTD", ampLTD);
      tauLTP = params->value(getName(), "tauLTP", tauLTP);
      tauLTD = params->value(getName(), "tauLTD", tauLTD);

      wMax = params->value(getName(), "wMax", wMax);
      wMin = params->value(getName(), "wMin", wMin);
      dWMax = params->value(getName(), "dWMax", dWMax);

   }

   return 0;
}

int STDPConn::updateState(float time, float dt)
{
   update_timer->start();

   int status=0;
   if (stdpFlag) {
      const float fac = ampLTD;
      const float decay = exp(-dt / tauLTD);

      //
      // both pDecr and activity are extended regions (plus margins)
      // to make processing them together simpler

      const int nk = pDecr->numItems;
      const float * a = post->getLayerData();
      float * m = pDecr->data; // decrement (minus) variable

      for (int k = 0; k < nk; k++) {
         m[k] = decay * m[k] - fac * a[k];
      }

      //const int axonId = 0;       // assume only one for now
      for(int axonId = 0; axonId<numberOfAxonalArborLists(); axonId++) {
         status=updateWeights(axonId);
      }
   }
   update_timer->stop();

   //const int axonId = 0;       // assume only one for now
   //return updateWeights(axonId);
   return status;
}

/**
 *  M (m or pDecr->data) is an extended post-layer variable
 *
 */
int STDPConn::updateWeights(int axonId)
{
   // Steps:
   // 1. Update pDecr (assume already done as it should only be done once)
   // 2. update Psij (dwPatches) for each synapse
   // 3. update wij

   const float dt = parent->getDeltaTime();
   const float decayLTP = exp(-dt / tauLTP);

   const int numExtended = pre->getNumExtended();
   assert(numExtended == getNumWeightPatches());

   const pvdata_t * preLayerData = pre->getLayerData();

   // this stride is in extended space for post-synaptic activity and
   // STDP decrement variable
   const int postStrideY = post->getLayerLoc()->nf
                         * (post->getLayerLoc()->nx + 2 * post->getLayerLoc()->nb);

   for (int kPre = 0; kPre < numExtended; kPre++) {
      // PVAxonalArbor * arbor = axonalArbor(kPre, axonId);

      const pvdata_t preActivity = preLayerData[kPre];

      // PVPatch * pIncrPatch   = dwPatches[axonId][kPre];
      PVPatch * w       = getWeights(kPre, axonId);
      size_t postOffset = getAPostOffset(kPre, axonId);

      const pvdata_t * postActivity = &post->getLayerData()[postOffset];
      const pvdata_t * M = &pDecr->data[postOffset];  // STDP decrement variable
      pvdata_t * P = get_dwData(axonId, kPre);        // STDP increment variable
      pvdata_t * W = get_wData(axonId, kPre); // w->data;

      int nk  = nfp * w->nx; // one line in x at a time
      int ny  = w->ny;
      int sy  = syp;

      // TODO - unroll

      // update Psij (dwPatches variable)
      // we are processing patches, one line in y at a time
      for (int y = 0; y < ny; y++) {
         pvpatch_update_plasticity_incr(nk, P + y * sy, preActivity, decayLTP, ampLTP);
      }

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
      assert(status == PV_SUCCESS);
   }
   else if ( (time >= writeTime) && (writeStep >= 0) ) {
      convertPreSynapticWeights(time);
      status = writePostSynapticWeights(time, last);
      assert(status == PV_SUCCESS);
   }

   return status;
}

float STDPConn::maxWeight(int arborID)
{
   return wMax;
}

int STDPConn::writeTextWeightsExtra(FILE * fd, int k, int arborID)
{
   if (stdpFlag) {
      pv_text_write_patch(fd, getWeights(k, arborID), get_dwData(arborID, k), nfp, sxp, syp, sfp); // write the Ps variable
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
#endif // NOTYET - TODO


int STDPConn::pvpatch_update_plasticity_incr(int nk, float * RESTRICT p,
                                   float aPre, float decay, float ltpAmp)
{
   int k;
   for (k = 0; k < nk; k++) {
      p[k] = decay * p[k] + ltpAmp * aPre;
   }
   return 0;
}

int STDPConn::pvpatch_update_weights(int nk, float * RESTRICT w, const float * RESTRICT m,
                           const float * RESTRICT p, float aPre,
                           const float * RESTRICT aPost, float dWMax, float wMin, float wMax)
{
   int k;
   for (k = 0; k < nk; k++) {
      // The next statement allows some synapses to "die".
      // TODO - check to see if its faster to not use branching
      if (w[k] < WEIGHT_MIN_VALUE) continue;
       w[k] += dWMax * (aPre * m[k] + aPost[k] * p[k]);
       w[k] = w[k] < wMin ? wMin : w[k];
       w[k] = w[k] > wMax ? wMax : w[k];
   }
   return 0;
}

#ifdef OBSOLETE // Marked obsolete Aug 29, 2011.  No function calls pvpatch_update_weights_localWMax
int STDPConn::pvpatch_update_weights_localWMax(int nk, float * RESTRICT w, const float * RESTRICT m,
                           const float * RESTRICT p, float aPre,
                           const float * RESTRICT aPost, float dWMax, float wMin, float * RESTRICT Wmax)
{
   int k;
   for (k = 0; k < nk; k++) {
      //printf("Wmax[%d] = %f m[%d] = %f\n",k,Wmax[k],k,m[k]);
      // The next statement allows some synapses to "die".
      // TODO - check to see if its faster to not use branching
      if (w[k] < WEIGHT_MIN_VALUE) continue;
       w[k] += dWMax * (aPre * m[k] + aPost[k] * p[k]);
       w[k] = w[k] < wMin ? wMin : w[k];
       w[k] = w[k] > Wmax[k] ? Wmax[k] : w[k];
   }
   return 0;
}
#endif // OBSOLETE

} // End of namespace PV

