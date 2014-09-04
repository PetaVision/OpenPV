/*
 * InhibSTDPConn.cpp
 *
 *  Created on: Mar 21, 2013
 *      Author: dpaiton
 */

#include "InhibSTDPConn.hpp"
#include "../layers/LIF.hpp"
#include "../io/io.h"
#include <assert.h>
#include <math.h>

namespace PV {

InhibSTDPConn::InhibSTDPConn(const char * name, HyPerCol * hc)
{
   initialize(name, hc);
}

int InhibSTDPConn::initialize(const char * name, HyPerCol * hc)
{
   int status = OjaSTDPConn::initialize(name, hc);
   return status;
}

int InhibSTDPConn::ioParamsFillGroup(enum ParamsIOFlag ioFlag)
{
   OjaSTDPConn::ioParamsFillGroup(ioFlag);
   return 0;
}

int InhibSTDPConn::updateWeights(int arborID)
{
   // Steps:
   // 2. Update pre_stdp_tr[arborID]
   // 3. Update w_ij

   const float dt                = parent->getDeltaTime();
   const float decayLTP          = exp(-dt / tauLTP);

   //Extended Pre
   const int nkPre = pre->getNumExtended();
   assert(nkPre == getNumWeightPatches());


   //Restricted Post
   const pvdata_t * aPost = post->getLayerData(getDelay(arborID));
   pvdata_t * post_stdp_tr_m;   // Postsynaptic trace matrix; i.e. data of post_stdp_tr struct
   pvdata_t * post_oja_tr_m;    // Postsynaptic mean trace matrix
   pvdata_t * ampLTD_m;         // local ampLTD

   //Extended Pre
   const pvdata_t * preLayerData = pre->getLayerData(getDelay(arborID));
   pvdata_t aPre;
   pvdata_t * pre_stdp_tr_m;    // Presynaptic trace matrix
   pvwdata_t * W;               // Weight matrix pointer

   //Restricted post vals
   const int postNx = post->getLayerLoc()->nx;
   const int postNy = post->getLayerLoc()->ny;
   const int postNf = post->getLayerLoc()->nf;
   const PVHalo * postHalo = &post->getLayerLoc()->halo;

   //stride in restricted space
   const int postStrideYRes = postNf * postNx;

   int nk, ny;

#ifdef SPLIT_PRE_POST //Separate LTD and LTP calculations to take advantage of sparsity
   //Loop over postsynaptic neurons for post before pre (tau'')
   const int numPostPatch = nxpPost * nypPost * nfpPost; // Number of pre-neurons in post receptive field. Postsynaptic weights are never shrunken

   //Update all pre traces (all traces decay every time step)
   for (int kPreExt = 0; kPreExt < nkPre; kPreExt++)           // Loop over all presynaptic neurons
   {
      aPre           = preLayerData[kPreExt];                  // Spiking activity
      pre_stdp_tr_m  = &(pre_stdp_tr[arborID]->data[kPreExt]); // PreTrace for given presynaptic neuron kPreExt
      *pre_stdp_tr_m = decayLTP * ((*pre_stdp_tr_m) + aPre);   //If spiked, minimum is 1. If no spike, minimum is 0.
   }

   pvwdata_t * startAdd = this->get_wDataStart(arborID); // Address of first neuron in pre layer
   //Loop through postsynaptic neurons (non-extended indices)
   for (int kPost = 0; kPost < post->getNumNeurons(); kPost++) { //Neuron indices
	   //Post in extended space
	   if (aPost[kPost] == 0) { //No LTP if post does not spike
		   continue;
	   }

	   pvdata_t ** postData = getPostWeightsp(arborID,kPost); // Pointer array full of addresses pointing to the weights for all of the preNeurons connected to the given postNeuron's receptive field
	   for (int kPrePatch=0; kPrePatch < numPostPatch; kPrePatch++) { // Loop through all pre-neurons connected to given post-neuron
		   float * kPreAdd = postData[kPrePatch];  // Address of preNeuron in receptive field of postNeuron
		   assert(kPreAdd != NULL);

           int kPreExt = (kPreAdd-startAdd) / (this->xPatchSize()*this->yPatchSize()*this->fPatchSize()); // Grab index based on patch size
           assert(kPreExt < nkPre);

		   // Pre in extended space
		   pre_stdp_tr_m  = &(pre_stdp_tr[arborID]->data[kPreExt]); // PreTrace for given presynaptic neuron kPreExt

		   // See STDP_LCA_Equations.pdf in documentation for description of feed-forward inhibitory weight adaptation equations. TODO: That file does not exist.
		   //STDP Equation
           (*postData[kPrePatch]) -= dWMax * ampLTP * aPost[kPost] * (*pre_stdp_tr_m);

           (*postData[kPrePatch]) = (*postData[kPrePatch]) < wMin ? wMin : (*postData[kPrePatch]); // Stop weights from going all the way to 0
	   }
   }

   // Pre-synaptic neurons for Pre Before Post (tau')
   for (int kPreExt = 0; kPreExt < nkPre; kPreExt++) // Loop over all presynaptic neurons
   {
      //Pre in extended space
      aPre = preLayerData[kPreExt]; // Spiking activity
      if (aPre == 0) { //No LTD if pre does not spike
    	  continue;
      }

      size_t postOffsetExt = getAPostOffset(kPreExt, arborID); // Gets start index for postsynaptic vectors for given presynaptic neuron and axon
      // size_t postOffsetRes = postOffsetExt - (postNb * (postNx + 2*postNb) + postNb);
      size_t postOffsetRes = kIndexRestricted(postOffsetExt, postNx, postNy, postNf, postNb);
      //Post in restricted space
      post_stdp_tr_m = &(post_stdp_tr->data[postOffsetRes]);   // Reference to STDP post trace (local)
      ampLTD_m       = &(ampLTD[postOffsetRes]);               // Points to local address

      W = get_wData(arborID, kPreExt);                         // Pointer to data of given axon & presynaptic neuron

      // Get weights in form of a patch (nx,ny,nf)
      // nk and ny are the number of neurons connected to the given presynaptic neuron in the x*nfp and y
      // if each of the presynaptic neurons connects to all postsynaptic than nk*ny = nkPost TODO: Is this true? Rui says yes.
      PVPatch * w = getWeights(kPreExt, arborID);                // Get weights in form of a patch (nx,ny,nf), TODO: what's the role of the offset?
      nk  = nfp * w->nx; // one line in x at a time
      ny  = w->ny;

      // 3. Update weights
      for (int y = 0; y < ny; y++) {
         for (int kPatchLoc = 0; kPatchLoc < nk; kPatchLoc++) { //loop over all postsynaptic neurons connected to given presynaptic neuron

            //STDP Equation
            W[kPatchLoc] += dWMax * ampLTD_m[kPatchLoc] * aPre * post_stdp_tr_m[kPatchLoc];

            W[kPatchLoc] = W[kPatchLoc] < wMin ? wMin : W[kPatchLoc]; // Stop weights from going all the way to 0
         }

         // advance pointers in y
         W += syp;

         // postActivity and post trace are extended layer
         post_stdp_tr_m += postStrideYRes;
         ampLTD_m       += postStrideYRes;
      }
   }
#else
   // this stride is in extended space for post-synaptic activity and STDP decrement variable
   const int postStrideYExt = postNf * (postNx + postHalo->lt + postHalo->rt);

   for (int kPreExt = 0; kPreExt < nkPre; kPreExt++)           // Loop over all presynaptic neurons
   {
      size_t postOffsetExt = getAPostOffset(kPreExt, arborID); // Gets start index for postsynaptic vectors for given presynaptic neuron and axon
      // size_t postOffsetRes = postOffsetExt - (postNb * (postNx + 2*postNb) + postNb);
      size_t postOffsetRes = kIndexRestricted(postOffsetExt, postNx, postNy, postNf, postHalo->lt, postHalo->rt, postHalo->dn, postHalo->up);

      //Post in extended space
      aPost          = &post->getLayerData()[postOffsetExt];   // Gets address of postsynaptic activity
      //Post in restricted space
      post_stdp_tr_m = &(post_stdp_tr->data[postOffsetRes]);   // Reference to STDP post trace (local)
      ampLTD_m       = &(ampLTD[postOffsetRes]);               // Points to local address
      //Pre in extended space
      aPre           = preLayerData[kPreExt];                  // Spiking activity
      pre_stdp_tr_m  = &(pre_stdp_tr[arborID]->data[kPreExt]); // PreTrace for given presynaptic neuron kPre

      W = get_wData(arborID, kPreExt);                         // Pointer to data of given axon & presynaptic neuron

      // Get weights in form of a patch (nx,ny,nf)
      // nk and ny are the number of neurons connected to the given presynaptic neuron in the x*nfp and y
      // if each of the presynaptic neurons connects to all postsynaptic than nk*ny = nkPost TODO: Is this true? Rui says yes.
      PVPatch * w = getWeights(kPreExt, arborID);                // Get weights in form of a patch (nx,ny,nf), TODO: what's the role of the offset?
      nk  = nfp * w->nx; // one line in x at a time
      ny  = w->ny;

      // 2. Updates the presynaptic trace
      *pre_stdp_tr_m = decayLTP * ((*pre_stdp_tr_m) + aPre);        //If spiked, minimum is 1. If no spike, minimum is 0.

      // 3. Update weights
      for (int y = 0; y < ny; y++) {
         for (int kPatchLoc = 0; kPatchLoc < nk; kPatchLoc++) { //loop over all postsynaptic neurons connected to given presynaptic neuron

            // See STDP_LCA_Equations.pdf in documentation for description of feed-forward inhibitory weight adaptation equations. TODO: That file does not exist.
            //STDP Equation
            W[kPatchLoc] += dWMax * (ampLTD_m[kPatchLoc] * aPre * post_stdp_tr_m[kPatchLoc] - ampLTP * aPost[kPatchLoc] * (*pre_stdp_tr_m));

            W[kPatchLoc] = W[kPatchLoc] < wMin ? wMin : W[kPatchLoc]; // Stop weights from going all the way to 0
         }

         // advance pointers in y
         W += syp;

         // postActivity and post trace are extended layer
         aPost          += postStrideYExt; //TODO: is this really in the extended space?
         post_stdp_tr_m += postStrideYRes;
         post_oja_tr_m  += postStrideYRes;
         ampLTD_m       += postStrideYRes;
      }
   }
#endif
   return 0;
}

} // End of namespace PV
