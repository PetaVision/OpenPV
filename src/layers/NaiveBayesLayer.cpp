/*
 * NaiveBayesLayer.cpp
 *
 *  Created on: Oct 29, 2012
 *      Author: garkenyon
 */

#include "NaiveBayesLayer.hpp"
#include "../columns/HyPerCol.hpp"
#include "HyPerLayer.hpp"

namespace PV {

NaiveBayesLayer::NaiveBayesLayer()
{
   initialize_base();
}

NaiveBayesLayer::NaiveBayesLayer(const char * name, HyPerCol * hc)
{
   initialize_base();
   initialize(name, hc);
}

NaiveBayesLayer::~NaiveBayesLayer()
{
}

int NaiveBayesLayer::initialize_base()
{
   numChannels = 3;
   return PV_SUCCESS;
}

int NaiveBayesLayer::initialize(const char * name, HyPerCol * hc)
{
   return HyPerLayer::initialize(name, hc);
}

int NaiveBayesLayer::allocateDataStructures() {
   int status = HyPerLayer::allocateDataStructures();
   inClassCount = (long *) calloc(this->getCLayer()->numNeurons, sizeof(long));
   assert(inClassCount != NULL);
   outClassCount = (long *) calloc(this->getCLayer()->numNeurons, sizeof(long));
   assert(outClassCount != NULL);
   inClassSum = (double *) calloc(this->getCLayer()->numNeurons, sizeof(double));
   assert(inClassSum != NULL);
   outClassSum = (double *) calloc(this->getCLayer()->numNeurons, sizeof(double));
   assert(outClassSum != NULL);
   return status;
}

int NaiveBayesLayer::updateState(double timef, double dt){

   return updateState(timef, dt, getLayerLoc(), getCLayer()->activity->data, getV(),
         getNumChannels(), GSyn[0], parent->columnId());
}


int NaiveBayesLayer::updateState(double timef, double dt, const PVLayerLoc * loc, pvdata_t * A,
         pvdata_t * V, int num_channels, pvdata_t * gSynHead, int columnID){
   pv_debug_info("[%d]: CliqueLayer::updateState:", columnID);

   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int num_neurons = nx * ny * nf;

   // Assumes that channels are contiguous in memory, i.e. GSyn[ch] = GSyn[0]+num_neurons*ch.  See allocateBuffers().
   pvdata_t * gSynExc = getChannelStart(gSynHead, CHANNEL_EXC, num_neurons);
   pvdata_t * gSynInh = getChannelStart(gSynHead, CHANNEL_INH, num_neurons);
   //pvdata_t * gSynInhB = getChannelStart(gSynHead, CHANNEL_INHB, num_neurons);

// assume bottomUp input to gSynExc, target lateral input to gSynInh, distractor lateral input to gSynInhB
   for (int kLocal = 0; kLocal < num_neurons; kLocal++) {
      pvdata_t bottomUp_input = gSynExc[kLocal];
      if (bottomUp_input <= 0.0f) {
         continue;
      }
      pvdata_t inClass_mask = gSynInh[kLocal];
      inClassCount[kLocal] += (inClass_mask > 0.0f);
      outClassCount[kLocal] += (inClass_mask <= 0.0f);
      inClassSum[kLocal] += (inClass_mask > 0.0f) * bottomUp_input;
      inClassSum[kLocal] += (inClass_mask <= 0.0f) * bottomUp_input;
      int kExtended = kIndexExtended(kLocal, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
      double log_prob = 0.0;
      if (inClassCount[kLocal]>0 && outClassCount[kLocal]>0){
         log_prob = log((inClassSum[kLocal]/inClassCount[kLocal])/(outClassSum[kLocal]/outClassCount[kLocal]));
         A[kExtended] = log_prob;
      }
      else if (inClassCount[kLocal]<0 && outClassCount[kLocal]==0){
         A[kExtended] = -FLT_MAX;
      }
      else if (inClassCount[kLocal]>0 && outClassCount[kLocal]==0){
         A[kExtended] = FLT_MAX;
      }
   } // k

   //resetGSynBuffers_HyPerLayer(num_neurons, getNumChannels(), gSynHead);
   // resetGSynBuffers();

   return 0;

   return PV_SUCCESS;
}

} // namespace PV
