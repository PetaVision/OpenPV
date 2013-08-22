/*
 * Triplet-based STDP with online implementation (the minimal version)
 * (see more at: Pfister et al. 2006)
 *
 *  Created on: Jul 24, 2012
 *      Author: Rui P. Costa
 *
 */

#include "STDP3Conn.hpp"
#include "../layers/LIF.hpp"
#include "../io/io.h"
#include <assert.h>
#include <math.h>

namespace PV {

STDP3Conn::STDP3Conn(const char * name, HyPerCol * hc,
      const char * pre_layer_name, const char * post_layer_name,
      const char * filename, bool stdpFlag,
      InitWeights *weightInit) : HyPerConn()
{
   initialize_base();
   initialize(name, hc, pre_layer_name, post_layer_name, filename, stdpFlag, weightInit);
}

STDP3Conn::~STDP3Conn()
{
   deleteWeights();
}

int STDP3Conn::initialize_base() {
   // Default STDP parameters for modifying weights; defaults are overridden in setParams().
   // this->dwPatches = NULL;
   this->post_tr = NULL;
   this->post2_tr = NULL;
   this->ampLTP = 0.0065;
   this->ampLTD = 0.0071;
   this->tauLTP = 16.8;
   this->tauLTD = 33.7;
   this->tauY = 114;
   this->dWMax = 1;
   this->synscalingFlag = false;
   this->synscaling_v = 1;

   return PV_SUCCESS;
}

int STDP3Conn::initialize(const char * name, HyPerCol * hc,
      const char * pre_layer_name, const char * post_layer_name,
      const char * filename, bool stdpFlag, InitWeights *weightInit)
{
   this->stdpFlag = stdpFlag; //needs to be before call to HyPerConn::initialize since it calls overridden methods that depend on stdpFlag being set.
   int status = HyPerConn::initialize(name, hc, pre_layer_name, post_layer_name, filename, weightInit);

   // status |= setParams(hc->parameters()); // called by HyPerConn::initialize since setParams is virtual
   // status |= initPlasticityPatches();     // called by HyPerConn::constructWeights since initPlasticityPatches is virtual

   // Moved to allocateDataStructures since point2PreSynapticWeights allocates post patches
   // if(synscalingFlag){
   //    point2PreSynapticWeights();
   // }
   status |= setParams(hc->parameters()); // needs to be called after HyPerConn::initialize since it depends on post being set

   return status;
}

int STDP3Conn::initPlasticityPatches()
{
   if (!stdpFlag) return PV_SUCCESS;

   int status = HyPerConn::initPlasticityPatches();
   assert(status == 0);
   //const int arbor = 0;
   //const int numAxons = numberOfAxonalArborLists();

   //   dwPatches = createWeights(NULL, numWeightPatches(), nxp, nyp, nfp, 0);
   post_tr = pvcube_new(&post->getCLayer()->loc, post->getNumExtended());
   post2_tr = pvcube_new(&post->getCLayer()->loc, post->getNumExtended());
   pre_tr = pvcube_new(&pre->getCLayer()->loc, pre->getNumExtended());
   assert(post_tr != NULL);
   assert(post2_tr != NULL);
   assert(pre_tr != NULL);

   return PV_SUCCESS;
}

int STDP3Conn::deleteWeights()
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
int STDP3Conn::initializeThreadBuffers(const char * kernelName)
{
   return 0;
}

int STDP3Conn::initializeThreadKernels(const char * kernelName)
{
   return 0;
}
#endif // PV_USE_OPENCL

PVLayerCube * STDP3Conn::getPlasticityDecrement()
{
   return post_tr;
}

// set member variables specified by user
/*
 * Using a dynamic_cast operator to convert (downcast) a pointer to a base class (HyPerLayer)
 * to a pointer to a derived class (LIF). This way I do not need to define a virtual
 * function getWmax() in HyPerLayer which only returns a NULL pointer in the base class.
 */
int STDP3Conn::setParams(PVParams * params)
{
   // stdpFlag is now set by constructor
   // stdpFlag = (bool) filep->value(getName(), "stdpFlag", (float) stdpFlag);
   HyPerConn::setParams(params);

   readAmpLTP(params);
   readAmpLTD(params);
   readTauLTP(params);
   readTauLTD(params);
   readTauY(params);
   readWMax(params);
   readWMin(params);
   read_dWMax(params);
   readSynscalingFlag(params);
   readSynscaling_v(params);

   //   if (stdpFlag) {
   //      ampLTP = params->value(getName(), "ampLTP", ampLTP);
   //      ampLTD = params->value(getName(), "ampLTD", ampLTD);
   //      tauLTP = params->value(getName(), "tauLTP", tauLTP);
   //      tauLTD = params->value(getName(), "tauLTD", tauLTD);
   //      tauY = params->value(getName(), "tauY", tauY);
   //
   //      wMax = params->value(getName(), "wMax", wMax);
   //      wMin = params->value(getName(), "wMin", wMin);
   //      // dWMax = params->value(getName(), "dWMax", dWMax); // dWMax is set in HyPerConn::setParams, called above.
   //      synscalingFlag = params->value(getName(), "synscalingFlag", synscalingFlag);
   //      synscaling_v = params->value(getName(), "synscaling_v", synscaling_v);
   //   }

   return 0;
}

void STDP3Conn::readAmpLTP(PVParams * params) {
   assert(!params->presentAndNotBeenRead(name, "stdpFlag"));
   if(stdpFlag) ampLTP = params->value(getName(), "ampLTP", ampLTP);
}

void STDP3Conn::readAmpLTD(PVParams * params) {
   assert(!params->presentAndNotBeenRead(name, "stdpFlag"));
   if(stdpFlag) ampLTD = params->value(getName(), "ampLTD", ampLTD);
}

void STDP3Conn::readTauLTP(PVParams * params) {
   assert(!params->presentAndNotBeenRead(name, "stdpFlag"));
   if(stdpFlag) tauLTP = params->value(getName(), "tauLTP", tauLTP);
}

void STDP3Conn::readTauLTD(PVParams * params) {
   assert(!params->presentAndNotBeenRead(name, "stdpFlag"));
   if(stdpFlag) tauLTD = params->value(getName(), "tauLTD", tauLTD);
}

void STDP3Conn::readTauY(PVParams * params) {
   assert(!params->presentAndNotBeenRead(name, "stdpFlag"));
   if(stdpFlag) tauY = params->value(getName(), "tauY", tauY);
}

void STDP3Conn::readWMax(PVParams * params) {
   assert(!params->presentAndNotBeenRead(name, "stdpFlag"));
   if(stdpFlag) wMax = params->value(getName(), "wMax", wMax);
}

void STDP3Conn::readWMin(PVParams * params) {
   assert(!params->presentAndNotBeenRead(name, "stdpFlag"));
   if(stdpFlag) wMin = params->value(getName(), "wMin", wMin);
}

void STDP3Conn::read_dWMax(PVParams * params) {
   assert(!params->presentAndNotBeenRead(name, "stdpFlag"));
   if(stdpFlag) HyPerConn::read_dWMax(params);
}

void STDP3Conn::readSynscalingFlag(PVParams * params) {
   assert(!params->presentAndNotBeenRead(name, "stdpFlag"));
   if(stdpFlag) synscalingFlag = params->value(getName(), "synscalingFlag", synscalingFlag);
}

void STDP3Conn::readSynscaling_v(PVParams * params) {
   assert(!params->presentAndNotBeenRead(name, "stdpFlag"));
   if(stdpFlag) synscaling_v = params->value(getName(), "synscaling_v", synscaling_v);
}

int STDP3Conn::allocateDataStructures() {
   HyPerConn::allocateDataStructures();
   if(synscalingFlag){
      point2PreSynapticWeights();
   }
   return PV_SUCCESS;
}

/**
 * First function to be executed
 * Updates the postsynaptic trace and calls the updateWeights function
 */
int STDP3Conn::updateState(double time, double dt)
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
 *  (see more at: Pfister et al. 2008)
 *
 */
int STDP3Conn::updateWeights(int axonId)
{
   // Steps:
   // 1. Update post_tr
   // 2. Update pre_tr
   // 3. Update w_ij

   const float dt = parent->getDeltaTime();
   const float decayLTP = exp(-dt / tauLTP);
   const float decayLTD = exp(-dt / tauLTD);
   const float decayY = exp(-dt / tauY);
   const int nkpre = pre->getNumExtended();
   assert(nkpre == getNumWeightPatches());
   const pvdata_t * preLayerData = pre->getLayerData();
   const pvdata_t * aPost = post->getLayerData();

   pvdata_t aPre;
   //PVPatch * w;

   pvdata_t * post_tr_m;
   pvdata_t * post2_tr_m;
   pvdata_t * pre_tr_m; // Presynaptic trace matrix
   pvdata_t * W;
   int nk, ny;

   const int nkPost = post_tr->numItems;
   post_tr_m = post_tr->data; // Postsynaptic trace matrix
   post2_tr_m = post2_tr->data; // Postsynaptic trace matrix


   // 1. Updates the postsynaptic traces
   for (int kPost = 0; kPost < nkPost; kPost++) {
      //post_tr_m[kPost] = (decayLTD * post_tr_m[kPost] + aPost[kPost]) > 1? 1 : (decayLTD * post_tr_m[kPost] + aPost[kPost]);
      post_tr_m[kPost] = decayLTD * post_tr_m[kPost] + aPost[kPost];
      post2_tr_m[kPost] = decayY * post2_tr_m[kPost] + aPost[kPost];
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
      post2_tr_m = &(post_tr->data[postOffset]);  // STDP y decrement variable
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

            W[k] += dWMax * (-ampLTD*aPre * post_tr_m[k] + ampLTP * aPost[k] * (pre_tr_m[0]*post2_tr_m[k]));

            W[k] = W[k] < wMin ? wMin : W[k];
            W[k] = W[k] > wMax ? wMax : W[k];

         }

         // advance pointers in y
         W += syp; //FIXME: W += nk
         //pre_tr_m += syp; //FIXME: pre_tr_m += syp;

         // postActivity and post trace are extended layer
         aPost += postStrideY; //TODO: is this really in the extended space?
         post_tr_m += postStrideY;
         post2_tr_m += postStrideY;
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

         // loop through post-synaptic neurons (non-extended indices)
         for (int kPost = 0; kPost < post_tr->numItems; kPost++) {

            pvdata_t ** postData = wPostDataStartp[axonID] + numPostPatch*kPost + 0;
            for (int kp = 0; kp < numPostPatch; kp++) {
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


int STDP3Conn::pvpatch_update_plasticity_incr(int nk, float * RESTRICT p,
      float aPre, float decay, float ltpAmp)
{
   int k;
   for (k = 0; k < nk; k++) {
      p[k] = decay * p[k] + ltpAmp * aPre;
   }
   return 0;
}


int STDP3Conn::pvpatch_update_weights(int nk, float * RESTRICT w, const float * RESTRICT m,
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


int STDP3Conn::outputState(double timef, bool last)
{
   int status;
   io_timer->start();

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

   // io timer already in HyPerConn::outputState, don't call twice
   io_timer->stop();

   status = HyPerConn::outputState(timef, last);

#ifdef OBSOLETE
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
#endif //OBSOLETE

   return status;
}

float STDP3Conn::maxWeight(int arborID)
{
   return wMax;
}

int STDP3Conn::writeTextWeightsExtra(PV_Stream * pvstream, int k, int arborID)
{
   if (stdpFlag) {
      pv_text_write_patch(pvstream, getWeights(k, arborID), get_dwData(arborID, k), nfp, sxp, syp, sfp); // write the Ps variable
   }
   return 0;
}

} // End of namespace PV

