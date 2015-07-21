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

STDPConn::STDPConn(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

STDPConn::~STDPConn()
{
   deleteWeights();
}

int STDPConn::initialize_base() {
   // Default STDP parameters for modifying weights; defaults are overridden in ioParams().
   // this->dwPatches = NULL;
   this->post_tr = NULL;
   this->ampLTP = 1.0;
   this->ampLTD = 1.1;
   this->tauLTP = 20;
   this->tauLTD = 20;
   this->dWMax = 1;
   this->synscalingFlag = false;
   this->synscaling_v = 1;

   return PV_SUCCESS;
}

int STDPConn::initialize(const char * name, HyPerCol * hc)
{
   int status = HyPerConn::initialize(name, hc);
   return status;
}

int STDPConn::allocateDataStructures() {
   HyPerConn::allocateDataStructures();
   if(synscalingFlag){
      point2PreSynapticWeights();
   }
   return PV_SUCCESS;
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

#ifdef PV_USE_OPENCL
int STDPConn::initializeThreadBuffers(const char * kernelName)
{
   return 0;
}

int STDPConn::initializeThreadKernels(const char * kernelName)
{
   return 0;
}
#endif // PV_USE_OPENCL

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
int STDPConn::ioParamsFillGroup(enum ParamsIOFlag ioFlag)
{
   HyPerConn::ioParamsFillGroup(ioFlag);
   ioParam_stdpFlag(ioFlag);
   ioParam_ampLTP(ioFlag);
   ioParam_ampLTD(ioFlag);
   ioParam_tauLTP(ioFlag);
   ioParam_tauLTD(ioFlag);
   ioParam_wMax(ioFlag);
   ioParam_wMin(ioFlag);
   ioParam_dWMax(ioFlag); // Function is defined in HyPerConn but HyPerConn doesn't use dWMax, so it is called by subclasses that need it.
   ioParam_synscalingFlag(ioFlag);
   ioParam_synscaling_v(ioFlag);

   return 0;
}

void STDPConn::ioParam_stdpFlag(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "stdpFlag", &stdpFlag, true/*default value*/, true/*warnIfAbsent*/);
}

void STDPConn::ioParam_ampLTP(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "stdpFlag"));
   if(stdpFlag) parent->ioParamValue(ioFlag, name, "ampLTP", &ampLTP, ampLTP);
}

void STDPConn::ioParam_ampLTD(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "stdpFlag"));
   if(stdpFlag) parent->ioParamValue(ioFlag, name, "ampLTD", &ampLTD, ampLTD);
}

void STDPConn::ioParam_tauLTP(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "stdpFlag"));
   if(stdpFlag)  parent->ioParamValue(ioFlag, name, "tauLTP", &tauLTP, tauLTP);
}

void STDPConn::ioParam_tauLTD(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "stdpFlag"));
   if(stdpFlag) parent->ioParamValue(ioFlag, name, "tauLTD", &tauLTD, tauLTD);
}

void STDPConn::ioParam_wMax(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "stdpFlag"));
   if(stdpFlag) parent->ioParamValue(ioFlag, name, "wMax", &wMax, wMax);
}

void STDPConn::ioParam_wMin(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "stdpFlag"));
   if(stdpFlag) parent->ioParamValue(ioFlag, name, "wMin", &wMin, wMin);
}

void STDPConn::ioParam_dwMax(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "stdpFlag"));
   if(stdpFlag) HyPerConn::ioParam_dWMax(ioFlag);
}

void STDPConn::ioParam_synscalingFlag(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "stdpFlag"));
   if(stdpFlag) parent->ioParamValue(ioFlag, name, "ampLTP", &ampLTP, ampLTP);
}

void STDPConn::ioParam_synscaling_v(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "stdpFlag"));
   if(stdpFlag) parent->ioParamValue(ioFlag, name, "ampLTP", &ampLTP, ampLTP);
}

/**
 * First function to be executed
 * Updates the postsynaptic trace and calls the updateWeights function
 */
int STDPConn::updateState(double time, double dt)
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
   const pvdata_t * preLayerData = pre->getLayerData(getDelay(axonId));
   const pvdata_t * aPost = post->getLayerData();

   pvdata_t aPre;
   //PVPatch * w;

   pvdata_t * post_tr_m;
   pvdata_t * pre_tr_m; // Presynaptic trace matrix
   pvwdata_t * W;
   int nk, ny;

   const int nkPost = post_tr->numItems;
   post_tr_m = post_tr->data; // Postsynaptic trace matrix


   // 1. Updates the postsynaptic traces
   for (int kPost = 0; kPost < nkPost; kPost++) {
             post_tr_m[kPost] = aPost[kPost] ? aPost[kPost] : (decayLTD * post_tr_m[kPost]);  //nearest neighbor approximation
            //post_tr_m[kPost] = decayLTD * post_tr_m[kPost] + aPost[kPost];
   }

   // this stride is in extended space for post-synaptic activity and STDP decrement variable
   const int postStrideY = post->getLayerLoc()->nf * (post->getLayerLoc()->nx + post->getLayerLoc()->halo.lt + post->getLayerLoc()->halo.rt);
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
         //pre_tr_m[0] = aPre ? aPre : (decayLTP * pre_tr_m[0]); // Commented because it does not account for previous time-step //nearest neighbor approximation
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

               pvwdata_t ** postData = wPostDataStartp[axonID] + numPostPatch*kPost + 0;
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

int STDPConn::outputState(double timef, bool last)
{
   int status;
   io_timer->start();

   if (last) {
      printf("Writing last STDP weights..%f\n",timef);
      convertPreSynapticWeights(timef);
      status = writePostSynapticWeights(timef, last);
      assert(status == 0);
   } else if ( (timef >= writeTime) && (writeStep >= 0) ) {
      //writeTime += writeStep; Done in HyperConn
      convertPreSynapticWeights(timef);
      status = writePostSynapticWeights(timef, false);
      assert(status == 0);

      // append to output file after original open
      //ioAppend = true;
   }

   // io timer already in HyPerConn::outputState, don't call twice
   io_timer->stop();

   status = HyPerConn::outputState(timef, last);

   return status;
}

float STDPConn::maxWeight(int arborID)
{
   return wMax;
}

int STDPConn::writeTextWeightsExtra(PV_Stream * pvstream, int k, int arborID)
{
   if (stdpFlag) {
      pv_text_write_patch(pvstream, getWeights(k, arborID), get_dwData(arborID, k), nfp, sxp, syp, sfp); // write the Ps variable
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

} // End of namespace PV

