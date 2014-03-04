/*
 * RescaleLayer.cpp
 *
 *  Created on: May 11, 2011
 *      Author: garkenyon
 */

#include "RescaleLayer.hpp"
#include <stdio.h>

#include "../include/default_params.h"

namespace PV {
RescaleLayer::RescaleLayer() {
   initialize_base();
}

RescaleLayer::RescaleLayer(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

RescaleLayer::~RescaleLayer()
{
   // Handled by CloneVLayer destructor
   // free(originalLayerName);
   // clayer->V = NULL;
}

int RescaleLayer::initialize_base() {
   originalLayer = NULL;
   targetMax = 1;
   targetMin = -1;
   targetMean = 0;
   targetStd = 1;
   rescaleMethod = NULL;
   return PV_SUCCESS;
}

int RescaleLayer::initialize(const char * name, HyPerCol * hc) {
   //int num_channels = sourceLayer->getNumChannels();
   int status_init = CloneVLayer::initialize(name, hc);

   return status_init;
}

int RescaleLayer::communicateInitInfo() {
   int status = CloneVLayer::communicateInitInfo();
   originalLayer = parent->getLayerFromName(originalLayerName);
   if (originalLayer==NULL) {
      fprintf(stderr, "Group \"%s\": Original layer \"%s\" must be a HyPer layer\n", name, originalLayerName);
   }
   return status;
}

// should inherit this method by default
// !!!Error??? CloneVLayer::allocateDataStructures() sets clayer->V = originalLayer->getV(), so free(clayer->V) below would seem to free V of original layer
// int RescaleLayer::allocateDataStructures() {
//    int status = CloneVLayer::allocateDataStructures();
//    free(clayer->V);
//    clayer->V = originalLayer->getV();

//    // Should have been initialized with zero channels, so GSyn should be NULL and freeChannels() call should be unnecessary
//    assert(GSyn==NULL);
//    // // don't need conductance channels
//    // freeChannels();

//    return status;
//}

int RescaleLayer::setParams(PVParams * params){
  //readOriginalLayerName(params);  // done in CloneVLayer
   CloneVLayer::setParams(params);
   readRescaleMethod(params);
   if (strcmp(rescaleMethod, "maxmin") == 0){
      readTargetMax(params);
      readTargetMin(params);
   }
   else if(strcmp(rescaleMethod, "meanstd") == 0){
      readTargetMean(params);
      readTargetStd(params);
   }
   else if(strcmp(rescaleMethod, "pointmeanstd") == 0){
      readTargetMean(params);
      readTargetStd(params);
   }
   else{
      fprintf(stderr, "Rescale Layer %s: rescaleMethod does not exist. Current implemented methods are maxmin, meanstd, pointmeanstd.\n",
            name);
      exit(PV_FAILURE);
   }
   return PV_SUCCESS;
}


void RescaleLayer::readTargetMax(PVParams * params){
   targetMax = params->value(name, "targetMax", targetMax);
}

void RescaleLayer::readTargetMin(PVParams * params){
   targetMin = params->value(name, "targetMin", targetMin);
}

void RescaleLayer::readTargetMean(PVParams * params){
   targetMean = params->value(name, "targetMean", targetMean);
}
void RescaleLayer::readTargetStd(PVParams * params){
   targetStd = params->value(name, "targetStd", targetStd);
}

void RescaleLayer::readRescaleMethod(PVParams * params){
   rescaleMethod = strdup(params->stringValue(name, "rescaleMethod", rescaleMethod));
}

int RescaleLayer::setActivity() {
   pvdata_t * activity = clayer->activity->data;
   memset(activity, 0, sizeof(pvdata_t) * clayer->numExtended);
   return 0;
}

  // GTK: changed to rescale activity instead of V
int RescaleLayer::updateState(double timef, double dt) {
   int status = PV_SUCCESS;

   //Check if an update is needed
   //TODO not needed anymore, done with needUpdate, although it would be nice to automatically trigger off orig layer
   if(checkIfUpdateNeeded()){
       int numNeurons = originalLayer->getNumNeurons();
       pvdata_t * A = clayer->activity->data;
       const pvdata_t * originalA = originalLayer->getCLayer()->activity->data;
       const PVLayerLoc * loc = getLayerLoc();
       const PVLayerLoc * locOriginal = originalLayer->getLayerLoc();
       //Make sure all sizes match
       //assert(locOriginal->nb == loc->nb);
       assert(locOriginal->nx == loc->nx);
       assert(locOriginal->ny == loc->ny);
       assert(locOriginal->nf == loc->nf);

       if (strcmp(rescaleMethod, "maxmin") == 0){
          float maxA = -1000000000;
          float minA = 1000000000;
          //Find max and min of A
          for (int k = 0; k < numNeurons; k++){
             int kextOriginal = kIndexExtended(k, locOriginal->nx, locOriginal->ny, locOriginal->nf, locOriginal->nb);
             if (originalA[kextOriginal] > maxA){
                maxA = originalA[kextOriginal];
             }
             if (originalA[kextOriginal] < minA){
                minA = originalA[kextOriginal];
             }
          }

#ifdef PV_USE_MPI
          MPI_Allreduce(MPI_IN_PLACE, &maxA, 1, MPI_FLOAT, MPI_MAX, parent->icCommunicator()->communicator());
          MPI_Allreduce(MPI_IN_PLACE, &minA, 1, MPI_FLOAT, MPI_MIN, parent->icCommunicator()->communicator());
#endif // PV_USE_MPI

          float rangeA = maxA - minA;
          for (int k = 0; k < numNeurons; k++){
             int kext = kIndexExtended(k, loc->nx, loc->ny, loc->nf, loc->nb);
             int kextOriginal = kIndexExtended(k, locOriginal->nx, locOriginal->ny, locOriginal->nf, locOriginal->nb);
             if (rangeA != 0){
                A[kext] = ((originalA[kextOriginal] - minA)/rangeA) * (targetMax - targetMin) + targetMin;
             }
             else{
                A[kext] = originalA[kextOriginal];
             }
          }
       }
       else if(strcmp(rescaleMethod, "meanstd") == 0){
          float sum = 0;
          float sumsq = 0;
          //Find sum of originalA
          for (int k = 0; k < numNeurons; k++){
             int kextOriginal = kIndexExtended(k, locOriginal->nx, locOriginal->ny, locOriginal->nf, locOriginal->nb);
             sum += originalA[kextOriginal];
          }

#ifdef PV_USE_MPI
          MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_FLOAT, MPI_SUM, parent->icCommunicator()->communicator());
#endif // PV_USE_MPI

          float mean = sum / originalLayer->getNumGlobalNeurons();

          //Find (val - mean)^2 of originalA
          for (int k = 0; k < numNeurons; k++){
             int kextOriginal = kIndexExtended(k, locOriginal->nx, locOriginal->ny, locOriginal->nf, locOriginal->nb);
             sumsq += (originalA[kextOriginal] - mean) * (originalA[kextOriginal] - mean);
          }

#ifdef PV_USE_MPI
          MPI_Allreduce(MPI_IN_PLACE, &sumsq, 1, MPI_FLOAT, MPI_SUM, parent->icCommunicator()->communicator());
#endif // PV_USE_MPI
          float std = sqrt(sumsq / originalLayer->getNumGlobalNeurons());
          //Normalize
          for (int k = 0; k < numNeurons; k++){
             int kext = kIndexExtended(k, loc->nx, loc->ny, loc->nf, loc->nb);
             int kextOriginal = kIndexExtended(k, locOriginal->nx, locOriginal->ny, locOriginal->nf, locOriginal->nb);
             if (std != 0){
                A[kext] = ((originalA[kextOriginal] - mean) * (targetStd/std) + targetMean);
             }
             else{
                A[kext] = originalA[kextOriginal];
             }
          }
       }
       else if(strcmp(rescaleMethod, "pointmeanstd") == 0){
          int nx = loc->nx;
          int ny = loc->ny;
          int nf = loc->nf;
          int nb = loc->nb;
          int nbOrig = locOriginal->nb;
          //Loop through all nx and ny
          for(int iY = 0; iY < ny; iY++){ 
             for(int iX = 0; iX < nx; iX++){ 
                //Find sum and sum sq in feature space
                float sum = 0;
                float sumsq = 0;
                for(int iF = 0; iF < nf; iF++){
                   int kextOriginal = kIndex(iX+nbOrig, iY+nbOrig, iF, nx+2*nbOrig, ny+2*nbOrig, nf);
                   sum += originalA[kextOriginal];
                }
                float mean = sum/nf;
                for(int iF = 0; iF < nf; iF++){
                   int kextOriginal = kIndex(iX+nbOrig, iY+nbOrig, iF, nx+2*nbOrig, ny+2*nbOrig, nf);
                   sumsq += (originalA[kextOriginal] - mean) * (originalA[kextOriginal] - mean);
                }
                float std = sqrt(sumsq/nf);
                for(int iF = 0; iF < nf; iF++){
                   int kextOriginal = kIndex(iX+nbOrig, iY+nbOrig, iF, nx+2*nbOrig, ny+2*nbOrig, nf);
                   int kext = kIndex(iX+nbOrig, iY+nbOrig, iF, nx+2*nb, ny+2*nb, nf);
                   if (std != 0){
                      A[kext] = ((originalA[kextOriginal] - mean) * (targetStd/std) + targetMean);
                   }
                   else{
                      A[kext] = originalA[kextOriginal];
                   }
                }
             }
          }
       }
       if( status == PV_SUCCESS ) status = updateActiveIndices();
       //Update lastUpdateTime
	   lastUpdateTime = parent->simulationTime();

   }
   return status;
}

bool RescaleLayer::checkIfUpdateNeeded() {
   bool needsUpdate = false;
   if (getPhase() > originalLayer->getPhase()) {
      needsUpdate = originalLayer->getLastUpdateTime() >= lastUpdateTime;
   }
   else {
      needsUpdate = originalLayer->getLastUpdateTime() > lastUpdateTime;
   }
   return needsUpdate;
}

} // end namespace PV

