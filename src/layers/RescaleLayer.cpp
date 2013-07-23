/*
 * RescaleLayer.cpp
 *
 *  Created on: May 11, 2011
 *      Author: garkenyon
 */

#include "HyPerLayer.hpp"
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
   free(originalLayerName);
   clayer->V = NULL;
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
   int status_init = HyPerLayer::initialize(name, hc, 0);

   // Moved to communicateInitInfo();
   // originalLayer = clone;
   // Moved to allocateDataStructures();
   // free(clayer->V);
   // clayer->V = sourceLayer->getV();
   //
   // // don't need conductance channels
   // freeChannels();

   return status_init;
}

int RescaleLayer::communicateInitInfo() {
   int status = HyPerLayer::communicateInitInfo();
   originalLayer = parent->getLayerFromName(originalLayerName);
   if (originalLayer==NULL) {
      fprintf(stderr, "Group \"%s\": Original layer \"%s\" must be a HyPer layer\n", name, originalLayerName);
   }
   return status;
}

int RescaleLayer::allocateDataStructures() {
   int status = HyPerLayer::allocateDataStructures();
   free(clayer->V);
   clayer->V = originalLayer->getV();

   // don't need conductance channels
   freeChannels();

   return status;
}

int RescaleLayer::setParams(PVParams * params){
   readOriginalLayerName(params);
   HyPerLayer::setParams(params);
   readRescaleMethod(params);
   if (strcmp(rescaleMethod, "maxmin") == 0){
      readTargetMax(params);
      readTargetMin(params);
   }
   else if(strcmp(rescaleMethod, "meanstd") == 0){
      readTargetMean(params);
      readTargetStd(params);
   }
   else{
      fprintf(stderr, "Rescale Layer %s: rescaleMethod does not exist. Current implemented methods are maxmin and meanstd.\n",
            name);
      exit(PV_FAILURE);
   }
   return PV_SUCCESS;
}

void RescaleLayer::readOriginalLayerName(PVParams * params) {
   const char * original_layer_name = params->stringValue(name, "originalLayerName");
   if( original_layer_name == NULL ) {
      fprintf(stderr, "RescaleLayer \"%s\": string parameter originalLayerName must be set\n", name);
      exit(EXIT_FAILURE);
   }
   originalLayerName = strdup(original_layer_name);
   if (originalLayerName==NULL) {
      fprintf(stderr, "RescaleLayer \"%s\" error: unable to copy originalLayerName \"%s\": %s\n", name, original_layer_name, strerror(errno));
      exit(EXIT_FAILURE);
   }
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

int RescaleLayer::updateState(double timef, double dt) {
   int status = PV_SUCCESS;
   int numNeurons = originalLayer->getNumNeurons();
   int kext;
   pvdata_t * V = clayer->V; 
   pvdata_t * A = getActivity();
   const PVLayerLoc * loc = getLayerLoc();
   const PVLayerLoc * sourceLoc = originalLayer->getLayerLoc();

   //Make sure layer loc and source layer loc is equivelent
   if (V == NULL){
      fprintf(stderr, "Rescale Layer %s: Source layer must have a V buffer to rescale. Exiting.\n",
            name);
      exit(PV_FAILURE);
   }
   assert(loc->nx == sourceLoc->nx);
   assert(loc->ny == sourceLoc->ny);
   assert(loc->nf == sourceLoc->nf);
   assert(loc->nb == sourceLoc->nb);

   if (strcmp(rescaleMethod, "maxmin") == 0){
      float maxV = -1000000000;
      float minV = 1000000000;
      //Find max and min of V
      for (int k = 0; k < numNeurons; k++){
         if (V[k] > maxV){
            maxV = V[k];
         }
         if (V[k] < minV){
            minV = V[k];
         }
      }

#ifdef PV_USE_MPI
      MPI_Allreduce(MPI_IN_PLACE, &maxV, 1, MPI_FLOAT, MPI_MAX, parent->icCommunicator()->communicator());
      MPI_Allreduce(MPI_IN_PLACE, &minV, 1, MPI_FLOAT, MPI_MIN, parent->icCommunicator()->communicator());
#endif // PV_USE_MPI

      float rangeV = maxV - minV;
      for (int k = 0; k < numNeurons; k++){
         kext = kIndexExtended(k, loc->nx, loc->ny, loc->nf, loc->nb);
         if (rangeV != 0){
            A[kext] = ((V[k] - minV)/rangeV) * (targetMax - targetMin) + targetMin;
         }
         else{
            A[kext] = V[k];
         }
      }
   }
   else if(strcmp(rescaleMethod, "meanstd") == 0){
      float sum = 0;
      float sumsq = 0;
      //Find sum of V
      for (int k = 0; k < numNeurons; k++){
         sum += V[k];
      }
#ifdef PV_USE_MPI
      MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_FLOAT, MPI_SUM, parent->icCommunicator()->communicator());
#endif // PV_USE_MPI

      float mean = sum / originalLayer->getNumGlobalNeurons();

      //Find (val - mean)^2 of V
      for (int k = 0; k < numNeurons; k++){
         sumsq += (V[k] - mean) * (V[k] - mean);
      }
#ifdef PV_USE_MPI
      MPI_Allreduce(MPI_IN_PLACE, &sumsq, 1, MPI_FLOAT, MPI_SUM, parent->icCommunicator()->communicator());
#endif // PV_USE_MPI
      float std = sqrt(sumsq / originalLayer->getNumGlobalNeurons());
      //Normalize
      for (int k = 0; k < numNeurons; k++){
         kext = kIndexExtended(k, loc->nx, loc->ny, loc->nf, loc->nb);
         if (std != 0){
            A[kext] = ((V[k] - mean) * (targetStd/std) + targetMean);
         }
         else{
            A[kext] = V[k];
         }
      }
   }
   if( status == PV_SUCCESS ) status = updateActiveIndices();
   return status;
}

} // end namespace PV

