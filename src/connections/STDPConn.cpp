/*
 * Standard pair-based STDP with online implementation
 * (see more at: http://www.scholarpedia.org/article/Spike-Timing_Dependent_Plasticity)
 *
 *  Created on: Jan 28, 2011
 *      Author: sorenrasmussen
 *
 *  Updated on: Jun 25, 2012
 *      Author: Rui P. Costa
 *
 *      Open questions:
 *      TODO: Implement STDP-kernel based version (should speed up as long as we have info about the spike times)
 */

#include "STDPConn.hpp"
#include "../layers/LIF.hpp"
#include "../io/io.h"
#include <assert.h>
#include <math.h>

namespace PV {

STDPConn::STDPConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
                   const char * filename, bool stdpFlag,
                   InitWeights *weightInit) : HyPerConn()
{
   initialize_base();
   initialize(name, hc, pre, post, filename, stdpFlag, weightInit);
}

STDPConn::~STDPConn()
{
   deleteWeights();
}

int STDPConn::initialize_base() {
   // Default STDP parameters for modifying weights; defaults are overridden in setParams().
   // this->dwPatches = NULL;
   this->post_tr = NULL;
   this->ampLTP = 1.0;
   this->ampLTD = 1.1;
   this->tauLTP = 20;
   this->tauLTD = 20;
   this->dWMax = 1;
   // TODO: Set the default values for wMin and wMax? Or are they already set somewhere?
   return PV_SUCCESS;
}

int STDPConn::initialize(const char * name, HyPerCol * hc,
                         HyPerLayer * pre, HyPerLayer * post,
                         const char * filename, bool stdpFlag, InitWeights *weightInit)
{
   this->stdpFlag = stdpFlag; //needs to be before call to HyPerConn::initialize since it calls overridden methods that depend on stdpFlag being set.
   int status = HyPerConn::initialize(name, hc, pre, post, filename, weightInit);

   status |= setParams(hc->parameters()); // needs to be called after HyPerConn::initialize since it depends on post being set
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
   post_tr = pvcube_new(&post->getCLayer()->loc, post->getNumExtended());
   pre_tr = pvcube_new(&pre->getCLayer()->loc, pre->getNumExtended());
   assert(post_tr != NULL);
   assert(pre_tr != NULL);

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
      pvcube_delete(post_tr);
      post_tr = NULL;
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
   return post_tr;
}

// set member variables specified by user
/*
 * Using a dynamic_cast operator to convert (downcast) a pointer to a base class (HyPerLayer)
 * to a pointer to a derived class (LIF). This way I do not need to define a virtual
 * function getWmax() in HyPerLayer which only returns a NULL pointer in the base class.
 */
int STDPConn::setParams(PVParams * params)
{
   // stdpFlag is now set by constructor
   // stdpFlag = (bool) filep->value(getName(), "stdpFlag", (float) stdpFlag);
   HyPerConn::setParams(params);

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

/**
 * First function to be executed
 * Updates the postsynaptic trace and calls the updateWeights function
 */
int STDPConn::updateState(float time, float dt)
{
   update_timer->start();

   int status=0;
   if (stdpFlag) {

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

#ifdef OBSOLETE
/**
 *  STDP online implementation
 *  (see more at: http://www.scholarpedia.org/article/Spike-Timing_Dependent_Plasticity)
 *
 *  M (m or post_tr->data) is an extended post-layer variable
 *
 */
int STDPConn::updateWeights(int axonId)
{
   // Steps:
   // 1. Update post_tr
   // 2. Update pre_tr
   // 3. Update w_ij

   const float dt = parent->getDeltaTime();
   const float decayLTP = exp(-dt / tauLTP);
   const float decayLTD = exp(-dt / tauLTD);
   const int numExtended = pre->getNumExtended();
   assert(numExtended == getNumWeightPatches());
   const pvdata_t * preLayerData = pre->getLayerData();

   const pvdata_t aPre;
   PVPatch * w;
   size_t postOffset;
   const pvdata_t * aPost;
   float pvdata_t * post_tr_m;
   pvdata_t * pre_tr_m;
   pvdata_t * W;
   int nk, ny, sy;
   int y, k;

   // Note: both post_tr and activity are extended regions (i.e. plus margins)
   // to make processing them together simpler
   //FIXME: But they shouldnt be, right? At least the weights in the margins shouldnt be updated.

   const int nkPost = post_tr->numItems;
   aPost = post->getLayerData();
   post_tr_m = post_tr->data; // Postsynaptic trace matrix

   // this stride is in extended space for post-synaptic activity and
   // STDP decrement variable
   const int postStrideY = post->getLayerLoc()->nf * (post->getLayerLoc()->nx + 2 * post->getLayerLoc()->nb);

   // 1. Updates the postsynaptic trace
   for (int kPost = 0; kPost < nkPost; kPost++) {
      post_tr_m[kPost] = decayLTD * post_tr_m[kPost] + aPost[kPost];
   }//TODO: Move this for loop inside the next for?

   for (int kPre = 0; kPre < numExtended; kPre++) {

      aPre = preLayerData[kPre];

      w = getWeights(kPre, axonId);
      postOffset = getAPostOffset(kPre, axonId); // TODO: what is this offset for?

      aPost = &post->getLayerData()[postOffset];
      post_tr_m = &post_tr->data[postOffset];  // STDP decrement variable (postsynaptic trace)
      pre_tr_m = get_dwData(axonId, kPre);        // STDP increment variable (presynaptic trace), Note: It uses dwData as the presynaptic trace variable, FIXME: Use an internal variable similar to post_tr?
      W = get_wData(axonId, kPre); // w->data; TODO: is this diff from w?

      nk  = nfp * w->nx; // one line in x at a time
      ny  = w->ny;
      sy  = syp;

      // update Psij (dwPatches variable)
      // we are processing patches, one line in y at a time
      for (y = 0; y < ny; y++) {
         pre_tr_m = pre_tr_m + y * sy;
         // 2. Updates the presynaptic trace
         for (k = 0; k < nk; k++) {
            pre_tr_m[k] = decayLTP * pre_tr_m[k] + aPre;
         }
      }

      //3. Update weights w_ij
      for (y = 0; y < ny; y++) {
         pvpatch_update_weights(nk, W, post_tr_m, pre_tr_m, aPre, aPost, dWMax, wMin, wMax);

         // advance pointers in y
         W += sy;
         pre_tr_m += sy;

         // postActivity and M are extended layer
         aPost += postStrideY;
         post_tr_m += postStrideY;
      }
   }

   return 0;
}
#endif //OBSOLETE

/**
 *  STDP online implementation
 *  (see more at: http://www.scholarpedia.org/article/Spike-Timing_Dependent_Plasticity)
 *
 */
int STDPConn::updateWeights(int axonId)
{
   // Steps:
   // 1. Update post_tr
   // 2. Update pre_tr
   // 3. Update w_ij

   const float dt = parent->getDeltaTime();
   const float decayLTP = exp(-dt / tauLTP);
   const float decayLTD = exp(-dt / tauLTD);
   const int nkpre = pre->getNumExtended();
   assert(nkpre == getNumWeightPatches());
   const pvdata_t * preLayerData = pre->getLayerData();
   const pvdata_t * aPost = post->getLayerData();

   pvdata_t aPre;
   //PVPatch * w;

   pvdata_t * post_tr_m;
   pvdata_t * pre_tr_m; // Presynaptic trace matrix
   pvdata_t * W;
   int nk, ny;

   const int nkPost = post_tr->numItems;
   post_tr_m = post_tr->data; // Postsynaptic trace matrix


   // 1. Updates the postsynaptic traces
   for (int kPost = 0; kPost < nkPost; kPost++) {
             //post_tr_m[kPost] = (decayLTD * post_tr_m[kPost] + aPost[kPost]) > 1? 1 : (decayLTD * post_tr_m[kPost] + aPost[kPost]);
            post_tr_m[kPost] = decayLTD * post_tr_m[kPost] + aPost[kPost];
   }

   // this stride is in extended space for post-synaptic activity and STDP decrement variable
   const int postStrideY = post->getLayerLoc()->nf * (post->getLayerLoc()->nx + 2 * post->getLayerLoc()->nb);
   //FIXME: In the first iteration post is -70!!

   for (int kPre = 0; kPre < nkpre; kPre++) {

         aPre = preLayerData[kPre];
         PVPatch * w = getWeights(kPre, axonId); //Get weights in form of a patch (nx,ny,nf), TODO: what's the role of the offset?
         size_t postOffset = getAPostOffset(kPre, axonId); //Gets start index for postsynaptic vectors given presynaptic neuron kPre

         aPost = &post->getLayerData()[postOffset]; //Gets postsynaptic activity
         post_tr_m = &(post_tr->data[postOffset]);  // STDP decrement variable
         //pre_tr_m = get_dwData(axonId, kPre);        // STDP increment variable
         pre_tr_m = &(pre_tr->data[kPre]);
         W = get_wData(axonId, kPre); // w->data;

         nk  = nfp * w->nx; // one line in x at a time
         ny  = w->ny;

         // 2. Updates the presynaptic trace
         //pre_tr_m[0] = (decayLTP * pre_tr_m[0] + aPre) > 1? 1 : (decayLTP * pre_tr_m[0] + aPre);
         pre_tr_m[0] = decayLTP * pre_tr_m[0] + aPre;

         //3. Update weights
         for (int y = 0; y < ny; y++) {
               for (int k = 0; k < nk; k++) {
                  // The next statement allows some synapses to "die".
                  if (W[k] < WEIGHT_MIN_VALUE) continue;

                   W[k] += dWMax * (-ampLTD*aPre * post_tr_m[k] + ampLTP * aPost[k] * pre_tr_m[0]);

                   W[k] = W[k] < wMin ? wMin : W[k];
                   W[k] = W[k] > wMax ? wMax : W[k];

               }

            // advance pointers in y
            W += syp; //FIXME: W += nk
            //pre_tr_m += syp; //FIXME: pre_tr_m += syp;

            // postActivity and post trace are extended layer
            aPost += postStrideY; //TODO: is this really in the extended space?
            post_tr_m += postStrideY;
         }

      }



//   for (int kPre = 0; kPre < nkpre; kPre++) {
//      w = getWeights(kPre, axonId);
//
//      pre_tr_m = get_dwData(axonId, kPre);  // STDP increment variable (presynaptic trace), Note: It uses dwData as the presynaptic trace variable, FIXME: Use an internal variable similar to post_tr?
//      W = get_wData(axonId, kPre); // w->data; TODO: is this diff from w?
//      aPre = preLayerData[kPre];
//
//      // 1. Updates the presynaptic trace
//      pre_tr_m[kPre] = decayLTP * pre_tr_m[kPre] + aPre;
//
//      for (int kPost = 0; kPost < nkPost; kPost++) {
//
//
//
//       // 3. Update weights w_ij
//       if (W[kPre] > WEIGHT_MIN_VALUE){
//        W[kPre] += dWMax * (-ampLTD*aPre * post_tr_m[kPre] + ampLTP * aPost[kPost] * pre_tr_m[kPre]);
//        W[kPre] = W[kPre] < wMin ? wMin : W[kPre];
//        W[kPre] = W[kPre] > wMax ? wMax : W[kPre];
//       }
//      }
   //}

   return 0;
}

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

int STDPConn::outputState(float timef, bool last)
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
#endif //NOTYET - TODO


#ifdef OBSOLETE
int STDPConn::pvpatch_update_plasticity_incr(int nk, float * RESTRICT p, float aPre, float decay, float ltpAmp)
{
   int k;
   //printf("nk = %u\n",nk);
   for (k = 0; k < nk; k++) {
      p[k] = decay * p[k] + ltpAmp * aPre;
   }
   return 0;
}

int STDPConn::pvpatch_update_weights(int nk, float * RESTRICT w, const float * RESTRICT m,
                           const float * RESTRICT p, float aPre,
                           float aPost, float dWMax, float wMin, float wMax)
{
   int k;
   for (k = 0; k < nk; k++) {
      // The next statement allows some synapses to "die".
      // TODO - check to see if its faster to not use branching
      if (w[k] < WEIGHT_MIN_VALUE) continue;
       w[k] += dWMax * (-ampLTD*aPre * m[k] + ampLTP * aPost[k] * p[k]);
      //w[k] += -ampLTD*aPre * m[k] + ampLTP * aPost[k] * p[k];
       w[k] = w[k] < wMin ? wMin : w[k];
       w[k] = w[k] > wMax ? wMax : w[k];
//       if((dWMax * (aPre * m[k] + aPost[k] * p[k]))>0){
//                printf("dw");
//       }
   }
   return 0;
}
#endif //OBSOLETE

} // End of namespace PV

