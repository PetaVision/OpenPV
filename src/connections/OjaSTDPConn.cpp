/*
 * OjaOjaSTDPConn.cpp
 *
 *  Created on: Sep 27, 2012
 *      Author: dpaiton
 */

#include "OjaSTDPConn.hpp"
#include "../layers/LIF.hpp"
#include "../io/io.h"
#include <assert.h>
#include <math.h>

namespace PV {

OjaSTDPConn::OjaSTDPConn(const char * name, HyPerCol * hc, HyPerLayer * pre,
      HyPerLayer * post, const char * filename, InitWeights *weightInit)
{
   initialize_base();
   initialize(name, hc, pre, post, filename, weightInit);
#ifdef PV_USE_OPENCL
   if(gpuAccelerateFlag)
      initializeGPU();
#endif
}

OjaSTDPConn::~OjaSTDPConn()
{
   free(ampLTD); ampLTD = NULL;
   deleteWeights();
}

int OjaSTDPConn::initialize_base() {
   // Default STDP parameters for modifying weights; defaults are overridden in setParams().
   this->post_stdp_tr   = NULL;
   this->post_oja_tr    = NULL;
   this->post_int_tr    = NULL;
   this->post_int_tr    = NULL;
   this->pre_stdp_tr    = NULL;
   this->pre_oja_tr     = NULL;

   this->ampLTP         = 0.357; //amp sets ratio of LTP to LTD, or how much more/less effective LTP is than LTD. LTP/LTD should ~= 0.9 per Gar
   this->tauLTP         = 16.8;
   this->tauLTD         = 33.7;
   this->tauOja         = 337;
   this->tauTHR         = 1000;
   this->weightDecay    = 0.01;
   this->dWMax          = 1;

   this->initAmpLTD     = 1;
   this->targetRateHz   = 1;
   this->LTDscale       = ampLTP;
   this->ampLTD         = NULL; // Will allocate later

   this->wMin           = 0.0001;
   this->wMax           = 1;

   this->ojaFlag        = true;
   this->synscalingFlag = false;
   this->synscaling_v   = 1;

   return PV_SUCCESS;
}

int OjaSTDPConn::initialize(const char * name, HyPerCol * hc,
      HyPerLayer * pre, HyPerLayer * post,
      const char * filename, InitWeights *weightInit)
{

   int status = HyPerConn::initialize(name, hc, pre, post, filename, weightInit);

   status |= initPlasticityPatches(); // needs to be called after HyPerConn::initialize since it depends on post being set

   point2PreSynapticWeights(); //set up post synaptic weight monitoring

   //Need to assert that the previous layer is a LIF layer.
   HyPerLayer * postHyPerLayer = this->postSynapticLayer();
   LIF * postLIF = NULL;
   postLIF = dynamic_cast <LIF*> (postHyPerLayer);
   assert(postLIF != NULL);

   //Grab VthRest from presynaptic LIF params
   float VthRest;
   VthRest = postLIF->getLIFParams()->VthRest;

   //allocate ampLTD and set to initial value
   ampLTD = (float *) calloc(post->getNumExtended(), sizeof(float));
   for (int kex = 0; kex < post->getNumExtended(); kex++) {
      ampLTD[kex] = initAmpLTD;
   }

   //set LTDscale param (should default to ampLTP
   LTDscale = hc->parameters()->value(name, "LTDscale", LTDscale);
   if (LTDscale < 0) {
      if (hc->columnId()==0) {
         fprintf(stderr,"OjaSTDPConn \"%s\": LTDscale must be positive (value in params is %f).\n", name, LTDscale);
      }
      abort();
   }

   return status;
}

// set member variables specified by user
/*
 * Using a dynamic_cast operator to convert (downcast) a pointer to a base class (HyPerLayer)
 * to a pointer to a derived class (LIF). This way I do not need to define a virtual
 * function getWmax() in HyPerLayer which only returns a NULL pointer in the base class.
 */
int OjaSTDPConn::setParams(PVParams * params)
{
   HyPerConn::setParams(params);

   ampLTP         = params->value(getName(), "ampLTP", ampLTP);
   initAmpLTD     = params->value(getName(), "initAmpLTD", initAmpLTD);
   tauLTP         = params->value(getName(), "tauLTP", tauLTP);
   tauLTD         = params->value(getName(), "tauLTD", tauLTD);
   tauOja         = params->value(getName(), "tauOja", tauOja);
   tauTHR         = params->value(getName(), "tauTHR", tauTHR);
   weightDecay    = params->value(getName(), "weightDecay", weightDecay);
   targetRateHz   = params->value(getName(), "targetRate", targetRateHz);

   tauO           = 1/(targetRateHz/1000); //I put this here because it could be a params file param at some point. - DMP

   wMax           = params->value(getName(), "wMax", wMax);
   wMin           = params->value(getName(), "wMin", wMin);
   dWMax          = params->value(getName(), "dWMax", dWMax);

   ojaFlag        = params->value(getName(), "ojaFlag",ojaFlag);
   synscalingFlag = params->value(getName(), "synscalingFlag", synscalingFlag);
   synscaling_v   = params->value(getName(), "synscaling_v", synscaling_v);

   return 0;
}

int OjaSTDPConn::initPlasticityPatches()
{
   if (!plasticityFlag) return PV_SUCCESS;

   int status = HyPerConn::initPlasticityPatches();
   assert(status == 0);

   post_stdp_tr = pvcube_new(&post->getCLayer()->loc, post->getNumExtended());
   post_oja_tr  = pvcube_new(&post->getCLayer()->loc, post->getNumExtended());
   post_int_tr  = pvcube_new(&post->getCLayer()->loc, post->getNumExtended());
   pre_stdp_tr  = pvcube_new(&pre->getCLayer()->loc, pre->getNumExtended());
   pre_oja_tr   = pvcube_new(&pre->getCLayer()->loc, pre->getNumExtended());

   assert(post_stdp_tr != NULL);
   assert(post_oja_tr  != NULL);
   assert(post_int_tr  != NULL);
   assert(pre_stdp_tr  != NULL);
   assert(pre_oja_tr   != NULL);

   int numPost = post_stdp_tr->numItems;
   int numPre  = pre_stdp_tr->numItems;
   for (int kexPost = 0; kexPost < numPost; kexPost++) {
      post_stdp_tr->data[kexPost] = tauLTD * targetRateHz/1000;
      post_oja_tr->data[kexPost]  = tauOja * targetRateHz/1000;
      post_int_tr->data[kexPost]  = tauO   * targetRateHz/1000;
   }
   for (int kexPre = 0; kexPre < numPre; kexPre++) {
      pre_stdp_tr->data[kexPre] = tauLTP * targetRateHz/1000;
      pre_oja_tr->data[kexPre]  = tauOja * targetRateHz/1000;
   }

   return PV_SUCCESS;
}

int OjaSTDPConn::initializeThreadBuffers() {return 0;}
int OjaSTDPConn::initializeThreadKernels() {return 0;}

int OjaSTDPConn::deleteWeights()
{
   if (!plasticityFlag) return PV_SUCCESS;

   pvcube_delete(post_stdp_tr);
   pvcube_delete(post_oja_tr);
   pvcube_delete(post_int_tr);
   pvcube_delete(pre_stdp_tr);
   pvcube_delete(pre_oja_tr);
   post_stdp_tr = NULL;
   post_oja_tr  = NULL;
   post_int_tr  = NULL;
   pre_stdp_tr  = NULL;
   pre_oja_tr   = NULL;

   return PV_SUCCESS; //HyPerConn destructor is automatically called
}

/**
 * First function to be executed
 * Updates the postsynaptic trace and calls the updateWeights function
 */
int OjaSTDPConn::updateState(float time, float dt)
{
   update_timer->start();

   int status=0;
   if (plasticityFlag) {
      for(int arborID = 0; arborID<numberOfAxonalArborLists(); arborID++) {
         status=updateWeights(arborID);
      }
   }

   update_timer->stop();

   return status;
}

int OjaSTDPConn::updateWeights(int arborID)
{
   // Steps:
   // 1. Update post_stdp_tr
   // 2. Update pre_stdp_tr
   // 3. Update w_ij

   const float dt            = parent->getDeltaTime();
   const float decayLTP      = exp(-dt / tauLTP);
   const float decayLTD      = exp(-dt / tauLTD);
   const float decayOja      = exp(-dt / tauOja);
   const float decayO        = exp(-dt / tauO);
   const float targetRatekHz = targetRateHz/1000; // Convert Hz to kHz

   const int nkPost = post_stdp_tr->numItems;
   const int nkPre  = pre->getNumExtended();
   assert(nkPre == getNumWeightPatches());

   const pvdata_t * preLayerData = pre->getLayerData(getDelay(arborID));
   const pvdata_t * aPost        = post->getLayerData();
   pvdata_t aPre;

   pvdata_t * post_stdp_tr_m;   // Postsynaptic trace matrix; i.e. data of post_stdp_tr struct
   pvdata_t * post_oja_tr_m;    // Postsynaptic mean trace matrix
   pvdata_t * post_int_tr_m;    // Postsynaptic mean trace matrix
   pvdata_t * pre_stdp_tr_m;    // Presynaptic trace matrix
   pvdata_t * pre_oja_tr_m;
   pvdata_t * W;                // Weight matrix pointer

   int nk, ny;

   post_stdp_tr_m = post_stdp_tr->data;
   post_oja_tr_m  = post_oja_tr->data;
   post_int_tr_m  = post_int_tr->data;

   // 1. Updates the postsynaptic traces
   for (int kPost = 0; kPost < nkPost; kPost++)
   {
      post_stdp_tr_m[kPost] = decayLTD * (post_stdp_tr_m[kPost] + aPost[kPost]);
      post_oja_tr_m[kPost]  = decayOja * (post_oja_tr_m[kPost] + aPost[kPost]);
      post_int_tr_m[kPost]  = decayO   * (post_int_tr_m[kPost] + aPost[kPost]);

      ampLTD[kPost] += (dt/tauTHR) * ((post_int_tr_m[kPost]/tauO) - targetRatekHz) * (LTDscale/targetRatekHz);
      ampLTD[kPost] = ampLTD[kPost] < 0 ? 0 : ampLTD[kPost]; // Stop ampLTD from being 0
      assert(ampLTD[kPost] == ampLTD[kPost]); // Make sure it is not NaN (which would only happen if tergetRatekHz or tauO becomes 0
   }

   // this stride is in extended space for post-synaptic activity and STDP decrement variable
   const int postStrideY = post->getLayerLoc()->nf * (post->getLayerLoc()->nx + 2 * post->getLayerLoc()->nb);
   //FIXME: In the first iteration post is -70!! (May not still be true)

   float scaleFactor;
   if (ojaFlag) {
      scaleFactor = dWMax * (dt / (tauOja * targetRatekHz));
   }
   else
   {
      scaleFactor = dWMax;
   }
   assert(scaleFactor == scaleFactor); // Make sure it is not NaN (can only happen if tauOja or targetRatekHz = 0)

   for (int kPre = 0; kPre < nkPre; kPre++)              // Loop over all presynaptic neurons
   {
      size_t postOffset = getAPostOffset(kPre, arborID);  // Gets start index for postsynaptic vectors for given presynaptic neuron and axon

      aPre           = preLayerData[kPre];                // Spiking activity
      aPost          = &post->getLayerData()[postOffset]; // Gets address of postsynaptic activity
      post_stdp_tr_m = &(post_stdp_tr->data[postOffset]); // Reference to STDP post trace
      post_oja_tr_m  = &(post_oja_tr->data[postOffset]);
      pre_stdp_tr_m  = &(pre_stdp_tr->data[kPre]);        // PreTrace for given presynaptic neuron kPre
      pre_oja_tr_m   = &(pre_oja_tr->data[kPre]);

      W = get_wData(arborID, kPre);                        // Pointer to data of given axon & presynaptic neuron

      // Get weights in form of a patch (nx,ny,nf)
      // nk and ny are the number of neurons connected to the given presynaptic neuron in the x*nfp and y
      // if each of the presynaptic neurons connects to all postsynaptic than nk*ny = nkPost TODO: Is this true? Rui says yes.
      PVPatch * w = getWeights(kPre, arborID);                // Get weights in form of a patch (nx,ny,nf), TODO: what's the role of the offset?
      nk  = nfp * w->nx; // one line in x at a time
      ny  = w->ny;

      // 2. Updates the presynaptic trace
      *pre_stdp_tr_m = decayLTP * ((*pre_stdp_tr_m) + aPre);        //If spiked, minimum is 1. If no spike, minimum is 0.
      *pre_oja_tr_m  = decayOja * ((*pre_oja_tr_m)  + aPre);

      //3. Update weights
      for (int y = 0; y < ny; y++) {
         for (int k = 0; k < nk; k++) {

            // See LCA_Equations.pdf in documentation for description of Oja (feed-forward weight adaptation) equations.
            float ojaTerm;
            if (ojaFlag) {
               ojaTerm = (post_oja_tr_m[k]/tauOja) * ((*pre_oja_tr_m/tauOja) - W[k] * (post_oja_tr_m[k]/tauOja));
               assert(ojaTerm == ojaTerm); // Make sure it is not NaN (only happens if tauOja is 0)
            } else { //should just be standard STDP at this point
              ojaTerm = 1.0;
            }

            W[k] += scaleFactor *
                  (ojaTerm * ampLTP * aPost[k] * (*pre_stdp_tr_m) - ampLTD[k] * aPre * post_stdp_tr_m[k] -
                  weightDecay * W[k]);
//            W[k] += scaleFactor * (ojaTerm  *
//                  (ampLTP * aPost[k] * (*pre_stdp_tr_m) - ampLTD[k] * aPre * post_stdp_tr_m[k]) -
//                  weightDecay * W[k]);

            W[k] = W[k] < wMin ? wMin : W[k]; // Stop weights from going all the way to 0
            if (!ojaFlag) { //oja term should get rid of the need to impose a maximum weight
               W[k] = W[k] > wMax ? wMax : W[k];
            }
         }

         // advance pointers in y
         W += syp; //FIXME: W += nk

         // postActivity and post trace are extended layer
         aPost          += postStrideY; //TODO: is this really in the extended space?
         post_stdp_tr_m += postStrideY;
         post_int_tr_m  += postStrideY;
         post_oja_tr_m  += postStrideY;
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

      for(int arborID=0;arborID<numberOfAxonalArborLists();arborID++) {
         //Loop through post-synaptic neurons (non-extended indices)
         for (int kPost = 0; kPost < post_stdp_tr->numItems; kPost++) {

            pvdata_t ** postData = wPostDataStartp[arborID] + numPostPatch*kPost + 0;
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

pvdata_t ** OjaSTDPConn::getPostWeights(int arborID, int kPost) {
   const int numPostPatch = nxpPost * nypPost * nfpPost; // Post-synaptic weights are never shrunken
   pvdata_t ** postData = wPostDataStartp[arborID] + numPostPatch*kPost + 0;

   return postData;
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


   return status;
}

float OjaSTDPConn::maxWeight(int arborID)
{
   return wMax;
}

int OjaSTDPConn::writeTextWeightsExtra(FILE * fd, int k, int arborID)
{
   if (plasticityFlag) {
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
   chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_pre_stdp_tr.pvp", cpDir, name);
   if (chars_needed >= PV_PATH_MAX) {
      fprintf(stderr, "LCALIFLateralConn::checkpointWrite error.  Path \"%s/%s_pre_stdp_tr.pvp\" is too long.\n", cpDir, name);
      abort();
   }
   read_pvdata(filename, parent->icCommunicator(), &timed, pre_stdp_tr->data, &loc, PV_FLOAT_TYPE, /*extended*/ false, /*contiguous*/ false);

   memcpy(&loc, post->getLayerLoc(), sizeof(PVLayerLoc));
   loc.nx += 2*loc.nb;
   loc.ny += 2*loc.nb;
   loc.nxGlobal = loc.nx * parent->icCommunicator()->numCommColumns();
   loc.nyGlobal = loc.ny * parent->icCommunicator()->numCommRows();
   loc.nb = 0;
   chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_post_stdp_tr.pvp", cpDir, name);
   if (chars_needed >= PV_PATH_MAX) {
      fprintf(stderr, "LCALIFLateralConn::checkpointWrite error.  Path \"%s/%s_post_stdp_tr.pvp\" is too long.\n", cpDir, name);
      abort();
   }
   read_pvdata(filename, parent->icCommunicator(), &timed, post_stdp_tr->data, &loc, PV_FLOAT_TYPE, /*extended*/ false, /*contiguous*/ false);

   if( (float) timed != *timef && parent->icCommunicator()->commRank() == 0 ) {
      fprintf(stderr, "Warning: %s and %s_A.pvp have different timestamps: %f versus %f\n", filename, name, (float) timed, *timef);
   }

   return status;
}

int OjaSTDPConn::checkpointWrite(const char * cpDir) {
   int status = HyPerConn::checkpointWrite(cpDir);
   // This is kind of hacky, but we save the extended buffers post_stdp_tr, pre_stdp_tr as if they were nonextended buffers of size (nx+2*nb)-by-(ny+2*nb)
   char filename[PV_PATH_MAX];
   int chars_needed;
   PVLayerLoc loc;

   memcpy(&loc, pre->getLayerLoc(), sizeof(PVLayerLoc));
   loc.nx += 2*loc.nb;
   loc.ny += 2*loc.nb;
   loc.nxGlobal = loc.nx * parent->icCommunicator()->numCommColumns();
   loc.nyGlobal = loc.ny * parent->icCommunicator()->numCommRows();
   loc.nb = 0;
   chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_pre_stdp_tr.pvp", cpDir, name);
   if (chars_needed >= PV_PATH_MAX) {
      fprintf(stderr, "LCALIFLateralConn::checkpointWrite error.  Path \"%s/%s_pre_stdp_tr.pvp\" is too long.\n", cpDir, name);
      abort();
   }
   write_pvdata(filename, parent->icCommunicator(), (double) parent->simulationTime(), pre_stdp_tr->data, &loc, PV_FLOAT_TYPE, /*extended*/ false, /*contiguous*/ false);

   memcpy(&loc, post->getLayerLoc(), sizeof(PVLayerLoc));
   loc.nx += 2*loc.nb;
   loc.ny += 2*loc.nb;
   loc.nxGlobal = loc.nx * parent->icCommunicator()->numCommColumns();
   loc.nyGlobal = loc.ny * parent->icCommunicator()->numCommRows();
   loc.nb = 0;
   chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_post_stdp_tr.pvp", cpDir, name);
   if (chars_needed >= PV_PATH_MAX) {
      fprintf(stderr, "LCALIFLateralConn::checkpointWrite error.  Path \"%s/%s_post_stdp_tr.pvp\" is too long.\n", cpDir, name);
      abort();
   }
   write_pvdata(filename, parent->icCommunicator(), (double) parent->simulationTime(), post_stdp_tr->data, &loc, PV_FLOAT_TYPE, /*extended*/ false, /*contiguous*/ false);
   return status;
}


} // End of namespace PV
