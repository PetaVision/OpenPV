/*
 * ShuffleLayer.cpp
 *
 *  Created on: May 11, 2011
 *      Author: garkenyon
 */

#include "HyPerLayer.hpp"
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

ShuffleLayer::~ShuffleLayer()
{
    clayer->V = NULL;
    if (indexArray != NULL){
       free(indexArray);
    }
}

int ShuffleLayer::initialize_base() {
   sourceLayer = NULL;
   shuffleMethod = NULL;
   return PV_SUCCESS;
}

int ShuffleLayer::initialize(const char * name, HyPerCol * hc) {
   //int num_channels = sourceLayer->getNumChannels();
   int status_init = HyPerLayer::initialize(name, hc, 0);

   // source layer name read in readOriginalLayer, called by setParams
   // sourceLayer = clone;

   // Moved to allocateDataStructures
   // // don't need conductance channels
   // freeChannels();
   //
   // int numGlobalNeurons = sourceLayer->getNumGlobalNeurons();
   //
   // indexArray = (int*) calloc(numGlobalNeurons, sizeof(int));
   return status_init;
}

int ShuffleLayer::setParams(PVParams * params){
   readOriginalLayerName(params);
   HyPerLayer::setParams(params);
   readShuffleMethod(params);
   //Read additional parameters based on shuffle method
   if (strcmp(shuffleMethod, "random") == 0){
   }
   else{
      fprintf(stderr, "Shuffle Layer: Shuffle method not recognized. Options are \"random\".\n");
      exit(PV_FAILURE);
   }
   return PV_SUCCESS;
}

void ShuffleLayer::readOriginalLayerName(PVParams * params) {
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

void ShuffleLayer::readShuffleMethod(PVParams * params){
   shuffleMethod = strdup(params->stringValue(name, "shuffleMethod", NULL));
}

int ShuffleLayer::allocateDataStructures() {
   int status = HyPerLayer::allocateDataStructures();

   // don't need conductance channels
   freeChannels();

   int numGlobalNeurons = sourceLayer->getNumGlobalNeurons();

   indexArray = (int*) calloc(numGlobalNeurons, sizeof(int));

   return status;
}

int ShuffleLayer::setActivity() {
   pvdata_t * activity = clayer->activity->data;
   memset(activity, 0, sizeof(pvdata_t) * clayer->numExtended);
   return 0;
}

void ShuffleLayer::randomShuffle(){
   int numGlobalNeurons = sourceLayer->getNumGlobalNeurons();

   //Take this out
   for (int i = 0; i < numGlobalNeurons; i++){
      indexArray[i] = i;
   }
   std::cout << numGlobalNeurons << "\n";
   //TODO:: make mapping of indexArray
}

int ShuffleLayer::updateState(double timef, double dt) {
   int status = PV_SUCCESS;
   int numNeurons = sourceLayer->getNumNeurons();
   int kext;
   //sourceData is extended
   const pvdata_t * sourceData = sourceLayer->getLayerData();
   const PVLayerLoc * loc = getLayerLoc();
   const PVLayerLoc * sourceLoc = sourceLayer->getLayerLoc();
	int comm_size = parent->icCommunicator()->commSize();
	int rank = parent->icCommunicator()->commRank();
   int rootproc = 0;
   
   //Make sure layer loc and source layer loc is equivelent
   assert(loc->nx == sourceLoc->nx);
   assert(loc->ny == sourceLoc->ny);
   assert(loc->nf == sourceLoc->nf);
   assert(loc->nb == sourceLoc->nb);
   
   //Create a one to one mapping of neuron to neuron
   if (rank == rootproc){
      if (strcmp(shuffleMethod, "random") == 0){
         randomShuffle();
      }
   }

   if( status == PV_SUCCESS ) status = updateActiveIndices();
   return status;
}

} // end namespace PV

