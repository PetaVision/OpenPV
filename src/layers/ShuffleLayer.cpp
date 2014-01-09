/*
 * ShuffleLayer.cpp
 *
 *  Created: July, 2013
 *   Author: Sheng Lundquist, Will Shainin
 */

#include "ShuffleLayer.hpp"
#include <stdio.h>

#include "../include/default_params.h"

namespace PV {
ShuffleLayer::ShuffleLayer() {
   initialize_base();
}

ShuffleLayer::ShuffleLayer(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

ShuffleLayer::~ShuffleLayer(){
   free(shuffleMethod);
   shuffleMethod = NULL;
}

int ShuffleLayer::initialize_base() {
   shuffleMethod = NULL;
   return PV_SUCCESS;
}

int ShuffleLayer::initialize(const char * name, HyPerCol * hc) {
   int status_init = HyPerLayer::initialize(name, hc, 0);
   // don't need conductance channels
   freeChannels(); // TODO: Does this need to be here?
   return status_init;
}

int ShuffleLayer::setParams(PVParams * params){
   int status = CloneVLayer::setParams(params);
   readShuffleMethod(params);
   //Read additional parameters based on shuffle method
   if (strcmp(shuffleMethod, "random") == 0){
   }
   else{
      fprintf(stderr, "Shuffle Layer: Shuffle method not recognized. Options are \"random\".\n");
      exit(PV_FAILURE);
   }
   return status;
}

void ShuffleLayer::readShuffleMethod(PVParams * params){
   shuffleMethod = strdup(params->stringValue(name, "shuffleMethod", false));
}

int ShuffleLayer::setActivity() {
   pvdata_t * activity = clayer->activity->data;
   memset(activity, 0, sizeof(pvdata_t) * clayer->numExtended);
   return 0;
}

void ShuffleLayer::randomShuffle(const pvdata_t * sourceData, pvdata_t * activity){
   const PVLayerLoc * loc = getLayerLoc();
   int nb    = loc->nb;
   int nxExt = loc->nx + 2*nb;
   int nyExt = loc->ny + 2*nb;
   int nf    = loc->nf;
   int numextended = getNumExtended();
   assert(numextended == nxExt * nyExt * nf);
   int rndIdx, rd;
   for (int i = 0; i < numextended; i++) { //Zero activity array for shuffling activity
      activity[i] = 0;
   }
   //NOTE: The following code assumes that the active features are sparse. 
   //      If the number of active features in sourceData is greater than 1/2 of nf, do..while will loop infinitely 
   
   for (int ky = 0; ky < nyExt; ky++){
      for (int kx = 0; kx < nxExt; kx++){
         for (int kf = 0; kf < nf; kf++){
            int extIdx = kIndex(kx, ky, kf, nxExt, nyExt, nf);
            float inData = sourceData[extIdx];
            if (inData != 0) { //Features with 0 activity are not changed
               do {
                  rd = rand() % nf; //TODO: Improve PRNG
                  rndIdx = kIndex(kx, ky, rd, nxExt, nyExt, nf);
               } while(sourceData[rndIdx] || activity[rndIdx]); 
               activity[rndIdx] = sourceData[extIdx];
               activity[extIdx] = sourceData[rndIdx];
            }
         }
      }
   }
}

int ShuffleLayer::updateState(double timef, double dt) {
   int status = PV_SUCCESS;
   int kext;
   //sourceData is extended
   const pvdata_t * sourceData = originalLayer->getLayerData();
   pvdata_t * A = getActivity();
   const PVLayerLoc * loc = getLayerLoc();
   const PVLayerLoc * sourceLoc = originalLayer->getLayerLoc();
	int comm_size = parent->icCommunicator()->commSize();
	int rank = parent->icCommunicator()->commRank();
   int rootproc = 0;
   
   //Make sure layer loc and source layer loc is equivelent
   assert(loc->nx == sourceLoc->nx);
   assert(loc->ny == sourceLoc->ny);
   assert(loc->nf == sourceLoc->nf);
   assert(loc->nb == sourceLoc->nb);
   
   //Create a one to one mapping of neuron to neuron
   if (strcmp(shuffleMethod, "random") == 0){
      randomShuffle(sourceData, A);
   }

   if( status == PV_SUCCESS ) status = updateActiveIndices();
   return status;
}

} // end namespace PV

