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
   free(featureFreqCount);
   shuffleMethod = NULL;
}

int ShuffleLayer::initialize_base() {
   shuffleMethod = NULL;
   maxCount = -99999999;
   freqCollectTime = 100;
   featureFreqCount = NULL;
   return PV_SUCCESS;
}

int ShuffleLayer::initialize(const char * name, HyPerCol * hc) {
   int status_init = HyPerLayer::initialize(name, hc, 0);
   // don't need conductance channels
   freeChannels(); // TODO: Does this need to be here?
   return status_init;
}

int ShuffleLayer::communicateInitInfo() {
   int status = CloneVLayer::communicateInitInfo();
   int nf = getLayerLoc()->nf;
   //Calloc to initialize all zeros
   featureFreqCount = (long*) calloc(nf, sizeof(long));
   return status;
}

int ShuffleLayer::setParams(PVParams * params){
   int status = CloneVLayer::setParams(params);
   readShuffleMethod(params);
   //Read additional parameters based on shuffle method
   if (strcmp(shuffleMethod, "random") == 0){
   }
   else if (strcmp(shuffleMethod, "rejection") == 0){
      readFreqCollectTime(params);
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

void ShuffleLayer::readFreqCollectTime(PVParams * params){
   freqCollectTime = params->value(name, "freqCollectTime", freqCollectTime);
}

int ShuffleLayer::setActivity() {
   pvdata_t * activity = clayer->activity->data;
   memset(activity, 0, sizeof(pvdata_t) * clayer->numExtended);
   return 0;
}

void ShuffleLayer::collectFreq(const pvdata_t * sourceData){
   int nb    = getLayerLoc()->nb;
   int nxExt = getLayerLoc()->nx + 2*nb;
   int nyExt = getLayerLoc()->ny + 2*nb;
   int nf    = getLayerLoc()->nf;
   for (int ky = 0; ky < nyExt; ky++){
      for (int kx = 0; kx < nxExt; kx++){
         for (int kf = 0; kf < nf; kf++){
            int extIdx = kIndex(kx, ky, kf, nxExt, nyExt, nf);
            float inData = sourceData[extIdx];
            //Really use 0? Or should there be a threshold parameter
            if(inData > 0){
               //Maybe do runing average, since this has a chance to overflow, but not likely
               featureFreqCount[kf]++;
            }
            if(featureFreqCount[kf] > maxCount){
               maxCount = featureFreqCount[kf];
            }
         }
      }
   }
#ifdef PV_USE_MPI
   //Collect over mpi
   MPI_Allreduce(MPI_IN_PLACE, featureFreqCount, nf, MPI_LONG, MPI_SUM, parent->icCommunicator()->communicator());
   MPI_Allreduce(MPI_IN_PLACE, &maxCount, 1, MPI_LONG, MPI_MAX, parent->icCommunicator()->communicator());
#endif // PV_USE_MPI
}

void ShuffleLayer::rejectionShuffle(const pvdata_t * sourceData, pvdata_t * activity){
   const PVLayerLoc * loc = getLayerLoc();
   int nb    = loc->nb;
   int nxExt = loc->nx + 2*nb;
   int nyExt = loc->ny + 2*nb;
   int nf    = loc->nf;
   int numextended = getNumExtended();
   assert(numextended == nxExt * nyExt * nf);
   int rndIdx, rdf;
   if(parent->simulationTime() <= freqCollectTime){
      //Collect maxVActivity and featureFreq
      collectFreq(sourceData);
   }
   else{
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
               //If no activity, reject and continue
               if(inData <= 0){
                  continue;
               }
               bool rejectFlag = true;
               while(rejectFlag){
                  //Grab random feature index
                  rdf = rand() % nf;
                  //Reject if random feature is itself
                  if(kf == rdf){
                     continue;
                  }
                  //Grab random index from 0 to 1
                  float prd = (float)rand() / (float)RAND_MAX;
                  //Compare frequency
                  if(prd < (float)featureFreqCount[rdf]/(float)maxCount){ 
                     //accepted
                     rejectFlag = false;
                  }
               }
               //rdf is now the random index to shuffle with
               rndIdx = kIndex(kx, ky, rdf, nxExt, nyExt, nf);
               activity[extIdx] = sourceData[rndIdx];
            }
         }
      }
   }
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
   //sourceData is extended
   const pvdata_t * sourceData = originalLayer->getLayerData();
   pvdata_t * A = getActivity();
   const PVLayerLoc * loc = getLayerLoc();
   const PVLayerLoc * sourceLoc = originalLayer->getLayerLoc();
   
   //Make sure layer loc and source layer loc is equivelent
   assert(loc->nx == sourceLoc->nx);
   assert(loc->ny == sourceLoc->ny);
   assert(loc->nf == sourceLoc->nf);
   assert(loc->nb == sourceLoc->nb);
   
   //Create a one to one mapping of neuron to neuron
   if (strcmp(shuffleMethod, "random") == 0){
      randomShuffle(sourceData, A);
   }
   else if(strcmp(shuffleMethod, "rejection") == 0){
      rejectionShuffle(sourceData, A);

   }

   if( status == PV_SUCCESS ) status = updateActiveIndices();
   return status;
}

} // end namespace PV

