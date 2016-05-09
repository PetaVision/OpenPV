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

   if(currFeatureFreqCount){
      free(currFeatureFreqCount[0]);
      free(currFeatureFreqCount);
      currFeatureFreqCount = NULL;
   }
   if(featureFreqCount){
      free(featureFreqCount[0]);
      free(featureFreqCount);
      featureFreqCount = NULL;
   }
   if(maxCount){
      free(maxCount);
   }

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
   //maxCount             = -99999999;
   maxCount             = NULL;
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

int ShuffleLayer::allocateDataStructures(){
   int status = CloneVLayer::allocateDataStructures();
   int nf = getLayerLoc()->nf;
   //Calloc to initialize all zeros
   featureFreqCount = (long**) calloc(getLayerLoc()->nbatch, sizeof(long*));
   long * tmp = (long*) calloc(getLayerLoc()->nbatch * nf, sizeof(long));
   assert(tmp);
   assert(featureFreqCount);
   for(int b = 0; b < getLayerLoc()->nbatch; b++){
      featureFreqCount[b] = &tmp[b*nf];
   }
   if (readFreqFromFile){
      readFreq();
   }
   else{
      currFeatureFreqCount = (long**) calloc(getLayerLoc()->nbatch, sizeof(long*));
      tmp = (long*) calloc(getLayerLoc()->nbatch * nf, sizeof(long));
      assert(tmp);
      assert(currFeatureFreqCount);
      for(int b = 0; b < getLayerLoc()->nbatch; b++){
         currFeatureFreqCount[b] = &tmp[b*nf];
      }
   }
   maxCount = (long*) malloc(getLayerLoc()->nbatch * sizeof(long));
   for(int b = 0; b < getLayerLoc()->nbatch; b++){
      maxCount[b] = -99999999;
   }
   return status;
}

int ShuffleLayer::communicateInitInfo() {
   int status = CloneVLayer::communicateInitInfo();
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
   memset(activity, 0, sizeof(pvdata_t) * clayer->numExtendedAllBatches);
   return 0;
}

void ShuffleLayer::readFreq(){ // TODO: Add MPI Bcast so that only root proc does this
   int nf = getLayerLoc()->nf;
   int nbatch = getLayerLoc()->nbatch;
   string line;
   ifstream freqFile(freqFilename);
   if (freqFile.is_open()){
      for (int b = 0; b < nbatch; b++){
         for (int kf = 0; kf < nf; kf++){
            getline (freqFile,line);
            if (freqFile.fail()){
               fprintf(stderr, "Shuffle Layer: Unable to read from frequency file %s\n",freqFilename);
               exit(PV_FAILURE);
            }
            featureFreqCount[b][kf] = atol(line.c_str());
            if(featureFreqCount[b][kf] > maxCount[b]){
               maxCount[b] = featureFreqCount[b][kf];
            }
            if (freqFile.eof()){
               fprintf(stderr, "Shuffle Layer: Invalid frequency file %s: EOF before %d nf, %d batches.\n ",freqFilename, nf, nbatch);
               exit(PV_FAILURE);
            }
         }
         if (getline(freqFile, line)){
               fprintf(stderr, "Shuffle Layer: Invalid frequency file: %s contains > %d nf, %d batches.\n ",freqFilename, nf, nbatch);
               exit(PV_FAILURE);
         }
      }
      freqFile.close();
   }
   else{
      fprintf(stderr, "Shuffle Layer: Unable to open frequency file %s\n",freqFilename);
      exit(PV_FAILURE);
   }
}

void ShuffleLayer::collectFreq(const pvdata_t * sourceData){
   PVHalo const * haloOrig = &originalLayer->getLayerLoc()->halo;
   int nx = getLayerLoc()->nx;
   int ny = getLayerLoc()->ny;
   int nxExt = nx + haloOrig->lt + haloOrig->rt;
   int nyExt = ny + haloOrig->dn + haloOrig->up;
   int nf    = getLayerLoc()->nf;
   int nbatch = getLayerLoc()->nbatch;
   for(int b = 0; b < nbatch; b++){
      const pvdata_t * sourceDataBatch = sourceData + b * nxExt * nyExt * nf;
      //Reset currFeatureFreqCount
      for(int kf = 0; kf < nf; kf++){
         currFeatureFreqCount[b][kf] = 0;
      }
      for (int ky = 0; ky < ny; ky++){
         for (int kx = 0; kx < nx; kx++){
            for (int kf = 0; kf < nf; kf++){
               int extIdx = kIndex(kx+haloOrig->lt, ky+haloOrig->up, kf, nxExt, nyExt, nf);
               float inData = sourceDataBatch[extIdx];
               if(inData > 0){   //Really use 0? Or should there be a threshold parameter
                  currFeatureFreqCount[b][kf]++;
               }
            }
         }
      }

      //Collect over mpi
      MPI_Allreduce(MPI_IN_PLACE, currFeatureFreqCount[b], nf, MPI_LONG, MPI_SUM, parent->icCommunicator()->communicator());
      
      for (int kf = 0; kf < nf; kf++){
         featureFreqCount[b][kf] += currFeatureFreqCount[b][kf];
         if(featureFreqCount[b][kf] > maxCount[b]){
            maxCount[b] = featureFreqCount[b][kf];
         }
      }
   }
}

void ShuffleLayer::rejectionShuffle(const pvdata_t * sourceData, pvdata_t * activity){
   const PVLayerLoc * loc = getLayerLoc();
   PVHalo const * haloOrig = &originalLayer->getLayerLoc()->halo;
   PVHalo const * halo = &loc->halo;
   int nx = loc->nx;
   int ny = loc->ny;
   int nf    = loc->nf;
   int numextended = getNumExtended();
   int nbatch = loc->nbatch;
   int rndIdx, rndIdxOrig;
   if(!readFreqFromFile && parent->simulationTime() <= freqCollectTime){
      //Collect maxVActivity and featureFreq
      collectFreq(sourceData);
   }
   else{
      for (int i = 0; i < numextended*nbatch; i++) { //Zero activity array for shuffling activity
         activity[i] = 0;
      }
      //NOTE: The following code assumes that the active features are sparse. 
      //      If the number of active features in sourceData is greater than 1/2 of nf, while will loop infinitely 
      for(int b = 0; b < nbatch; b++){
         pvdata_t * activityBatch = activity + b * numextended;
         const pvdata_t * sourceDataBatch = sourceData + b * numextended;
         for (int ky = 0; ky < ny; ky++){
            for (int kx = 0; kx < nx; kx++){
               int extIdx = kIndex(kx+halo->lt, ky+halo->up, 0, nx+halo->lt+halo->rt, ny+halo->dn+halo->up, nf);
               int extIdxOrig = kIndex(kx+haloOrig->lt, ky+haloOrig->up, 0, nx+haloOrig->lt+haloOrig->rt, ny+haloOrig->dn+haloOrig->up, nf);
               // Assumes stride in features is 1 when computing indices for features other than kf=0
               for (int kf = 0; kf < nf; kf++){
                  float inData = sourceDataBatch[extIdxOrig+kf];
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
                     if(sourceDataBatch[rndIdxOrig] || activityBatch[rndIdx]){
                        continue;
                     }
                     //Grab random index from 0 to 1
                     float prd = (float)rand() / (float)RAND_MAX;
                     //Compare frequency
                     if(prd <= (float)featureFreqCount[b][rdf]/(float)maxCount[b]){ 
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
}

void ShuffleLayer::randomShuffle(const pvdata_t * sourceData, pvdata_t * activity){
   const PVLayerLoc * loc = getLayerLoc();
   PVHalo const * haloOrig = &originalLayer->getLayerLoc()->halo;
   PVHalo const * halo = &loc->halo;
   int nx = loc->nx;
   int ny = loc->ny;
   int nf    = loc->nf;
   int nbatch = loc->nbatch;
   int numextended = getNumExtended();
   int rndIdx, rndIdxOrig;
   for (int i = 0; i < numextended * nbatch; i++) { //Zero activity array for shuffling activity
      activity[i] = 0;
   }
   //NOTE: The following code assumes that the active features are sparse. 
   //      If the number of active features in sourceData is greater than 1/2 of nf, do..while will loop infinitely 
   for(int b = 0; b < nbatch; b++){
      const pvdata_t * sourceDataBatch = sourceData + b * numextended;
      pvdata_t * activityBatch = activity + b * numextended;
      for (int ky = 0; ky < ny; ky++){
         for (int kx = 0; kx < nx; kx++){
            int extIdx = kIndex(kx+halo->lt, ky+halo->rt, 0, nx+halo->lt+halo->rt, ny+halo->dn+halo->up, nf);
            int extIdxOrig = kIndex(kx+haloOrig->lt, ky+haloOrig->up, 0, nx+haloOrig->lt+haloOrig->rt, ny+haloOrig->dn+haloOrig->up, nf);
            // Assumes stride in features is 1 when computing indices for features other than kf=0
            for (int kf = 0; kf < nf; kf++){
               float inData = sourceDataBatch[extIdxOrig+kf];
               if (inData != 0) { //Features with 0 activity are not changed
                  do {
                     int rd = rand() % nf; //TODO: Improve PRNG
                     rndIdx = extIdx + rd;
                     rndIdxOrig = extIdxOrig + rd;
                  } while(sourceDataBatch[rndIdxOrig] || activityBatch[rndIdx]); 
                  activityBatch[rndIdx] = sourceDataBatch[extIdxOrig+kf];
                  activityBatch[extIdx+kf] = sourceDataBatch[rndIdxOrig];
               }
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

   return status;
}

BaseObject * createShuffleLayer(char const * name, HyPerCol * hc) {
   return hc ? new ShuffleLayer(name, hc) : NULL;
}

} // end namespace PV

