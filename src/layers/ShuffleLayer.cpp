/*
 * ShuffleLayer.cpp
 *
 *  Created: July, 2013
 *   Author: Sheng Lundquist, Will Shainin
 */

#include "ShuffleLayer.hpp"
#include <stdio.h>

#include <iostream>
#include <fstream>
#include <string>
using namespace std;
//// DEBUG (REMOVE)

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
   free(freqFilename);
   free(featureFreqCount);
   free(currFeatureFreqCount);
   shuffleMethod        = NULL;
   freqFilename         = NULL;
   featureFreqCount     = NULL;
   currFeatureFreqCount = NULL;
}

int ShuffleLayer::initialize_base() {
   shuffleMethod        = NULL;
   freqFilename         = NULL;
   featureFreqCount     = NULL;
   currFeatureFreqCount = NULL;
   maxCount             = -99999999;
   freqCollectTime      = 1000;
   readFreqFromFile     = 0;
   return PV_SUCCESS;
}

int ShuffleLayer::initialize(const char * name, HyPerCol * hc) {
   int status_init = HyPerLayer::initialize(name, hc);
   // don't need conductance channels
   freeChannels(); // TODO: Does this need to be here?
   return status_init;
}

int ShuffleLayer::communicateInitInfo() {
   int status = CloneVLayer::communicateInitInfo();
   int nf = getLayerLoc()->nf;
   //Calloc to initialize all zeros
   featureFreqCount = (long*) calloc(nf, sizeof(long));
   if (readFreqFromFile){
      readFreq();
   }
   else{
      currFeatureFreqCount = (long*) calloc(nf, sizeof(long));
   }
   return status;
}

int ShuffleLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag){
   int status = CloneVLayer::ioParamsFillGroup(ioFlag);
   ioParam_shuffleMethod(ioFlag);
   ioParam_readFreqFromFile(ioFlag);
   ioParam_freqFilename(ioFlag);
   ioParam_freqCollectTime(ioFlag);
   return status;
}

void ShuffleLayer::ioParam_shuffleMethod(enum ParamsIOFlag ioFlag){
   parent->ioParamString(ioFlag, name, "shuffleMethod", &shuffleMethod, "random", false/*warnIfAbsent*/);
   //ioFlag==PARAMS_IO_READ && 
   if ((strcmp(shuffleMethod, "random") == 0 || strcmp(shuffleMethod, "rejection") == 0)){
   }
   else{
      fprintf(stderr, "Shuffle Layer: Shuffle method not recognized. Options are \"random\" or \"rejection\".\n");
      exit(PV_FAILURE);
   }
}

void ShuffleLayer::ioParam_readFreqFromFile(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "shuffleMethod"));
   if (strcmp(shuffleMethod, "rejection") == 0){
      parent->ioParamValue(ioFlag, name, "readFreqFromFile", &readFreqFromFile, readFreqFromFile);
   }
}

void ShuffleLayer::ioParam_freqFilename(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "readFreqFromFile"));
   if (readFreqFromFile){
      parent->ioParamString(ioFlag, name, "freqFilename", &freqFilename, freqFilename);
   }
}

void ShuffleLayer::ioParam_freqCollectTime(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "readFreqFromFile"));
   if (!readFreqFromFile){
      parent->ioParamValue(ioFlag, name, "freqCollectTime", &freqCollectTime, freqCollectTime);
   }
}

int ShuffleLayer::setActivity() {
   pvdata_t * activity = clayer->activity->data;
   memset(activity, 0, sizeof(pvdata_t) * clayer->numExtended);
   return 0;
}

void ShuffleLayer::readFreq(){ // TODO: Add MPI Bcast so that only root proc does this
   int nf = getLayerLoc()->nf;
   string line;
   ifstream freqFile(freqFilename);
   if (freqFile.is_open()){
      for (int kf = 0; kf < nf; kf++){
         getline (freqFile,line);
         if (freqFile.fail()){
            fprintf(stderr, "Shuffle Layer: Unable to read from frequency file %s\n",freqFilename);
            exit(PV_FAILURE);
         }
         featureFreqCount[kf] = atol(line.c_str());
         if(featureFreqCount[kf] > maxCount){
            maxCount = featureFreqCount[kf];
         }
         if (freqFile.eof()){
            fprintf(stderr, "Shuffle Layer: Invalid frequency file %s: EOF before %d nf.\n ",freqFilename, nf);
            exit(PV_FAILURE);
         }
      }
      if (getline(freqFile, line)){
            fprintf(stderr, "Shuffle Layer: Invalid frequency file: %s contains > %d nf.\n ",freqFilename, nf);
            exit(PV_FAILURE);
      }
      freqFile.close();
   }
   else{
      fprintf(stderr, "Shuffle Layer: Unable to open frequency file %s\n",freqFilename);
      exit(PV_FAILURE);
   }
}
void ShuffleLayer::collectFreq(const pvdata_t * sourceData){
   int nbOrig = originalLayer->getLayerLoc()->nb;
   int nx = getLayerLoc()->nx;
   int ny = getLayerLoc()->ny;
   int nxExt = nx + 2*nbOrig;
   int nyExt = ny + 2*nbOrig;
   int nf    = getLayerLoc()->nf;
   //Reset currFeatureFreqCount
   for(int kf = 0; kf < nf; kf++){
      currFeatureFreqCount[kf] = 0;
   }
   for (int ky = 0; ky < ny; ky++){
      for (int kx = 0; kx < nx; kx++){
         for (int kf = 0; kf < nf; kf++){
            int extIdx = kIndex(kx+nbOrig, ky+nbOrig, kf, nxExt, nyExt, nf);
            float inData = sourceData[extIdx];
            if(inData > 0){   //Really use 0? Or should there be a threshold parameter
               currFeatureFreqCount[kf]++;
            }
         }
      }
   }

#ifdef PV_USE_MPI
   //Collect over mpi
   MPI_Allreduce(MPI_IN_PLACE, currFeatureFreqCount, nf, MPI_LONG, MPI_SUM, parent->icCommunicator()->communicator());
#endif // PV_USE_MPI
   
   for (int kf = 0; kf < nf; kf++){
      featureFreqCount[kf] += currFeatureFreqCount[kf];
      if(featureFreqCount[kf] > maxCount){
         maxCount = featureFreqCount[kf];
      }
   }
}

void ShuffleLayer::rejectionShuffle(const pvdata_t * sourceData, pvdata_t * activity){
   const PVLayerLoc * loc = getLayerLoc();
   int nbOrig = originalLayer->getLayerLoc()->nb;
   int nb = loc->nb;
   int nx = loc->nx;
   int ny = loc->ny;
   int nf    = loc->nf;
   int numextended = getNumExtended();
   int rndIdx, rndIdxOrig;
   if(!readFreqFromFile && parent->simulationTime() <= freqCollectTime){
      //Collect maxVActivity and featureFreq
      collectFreq(sourceData);
   }
   else{
      for (int i = 0; i < numextended; i++) { //Zero activity array for shuffling activity
         activity[i] = 0;
      }
      //NOTE: The following code assumes that the active features are sparse. 
      //      If the number of active features in sourceData is greater than 1/2 of nf, while will loop infinitely 
      for (int ky = 0; ky < ny; ky++){
         for (int kx = 0; kx < nx; kx++){
            int extIdx = kIndex(kx+nb, ky+nb, 0, nx+2*nb, ny+2*nb, nf);
            int extIdxOrig = kIndex(kx+nbOrig, ky+nbOrig, 0, nx+2*nbOrig, ny+2*nbOrig, nf);
            // Assumes stride in features is 1 when computing indices for features other than kf=0
            for (int kf = 0; kf < nf; kf++){
               float inData = sourceData[extIdxOrig+kf];
               //If no activity, reject and continue
               if(inData <= 0){
                  continue;
               }
               bool rejectFlag = true;
               while(rejectFlag){
                  //Grab random feature index
                  int rdf = rand() % nf;
                  rndIdx = extIdx + rdf;
                  rndIdxOrig = extIdxOrig + rdf;
                  //Reject if random feature is active, or has been swapped
                  if(sourceData[rndIdxOrig] || activity[rndIdx]){
                     continue;
                  }
                  //Grab random index from 0 to 1
                  float prd = (float)rand() / (float)RAND_MAX;
                  //Compare frequency
                  if(prd <= (float)featureFreqCount[rdf]/(float)maxCount){ 
                     //accepted
                     rejectFlag = false;
                  }
               }
               //rdf is now the random index to shuffle with
               activity[rndIdx] = sourceData[extIdxOrig+kf];
               activity[extIdx+kf] = sourceData[rndIdxOrig]; // << This doesn't do anything
            }
         }
      }
   }
}

void ShuffleLayer::randomShuffle(const pvdata_t * sourceData, pvdata_t * activity){
   const PVLayerLoc * loc = getLayerLoc();
   int nbOrig = originalLayer->getLayerLoc()->nb;
   int nb = loc->nb;
   int nx = loc->nx;
   int ny = loc->ny;
   int nf    = loc->nf;
   int numextended = getNumExtended();
   int rndIdx, rndIdxOrig;
   for (int i = 0; i < numextended; i++) { //Zero activity array for shuffling activity
      activity[i] = 0;
   }
   //NOTE: The following code assumes that the active features are sparse. 
   //      If the number of active features in sourceData is greater than 1/2 of nf, do..while will loop infinitely 
   
   for (int ky = 0; ky < ny; ky++){
      for (int kx = 0; kx < nx; kx++){
         int extIdx = kIndex(kx+nb, ky+nb, 0, nx+2*nb, ny+2*nb, nf);
         int extIdxOrig = kIndex(kx+nbOrig, ky+nbOrig, 0, nx+2*nbOrig, ny+2*nbOrig, nf);
         // Assumes stride in features is 1 when computing indices for features other than kf=0
         for (int kf = 0; kf < nf; kf++){
            float inData = sourceData[extIdxOrig+kf];
            if (inData != 0) { //Features with 0 activity are not changed
               do {
                  int rd = rand() % nf; //TODO: Improve PRNG
                  rndIdx = extIdx + rd;
                  rndIdxOrig = extIdxOrig + rd;
               } while(sourceData[rndIdxOrig] || activity[rndIdx]); 
               activity[rndIdx] = sourceData[extIdxOrig+kf];
               activity[extIdx+kf] = sourceData[rndIdxOrig];
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
   //assert(loc->nb == sourceLoc->nb);
   
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

