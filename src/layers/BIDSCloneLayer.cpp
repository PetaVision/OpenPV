/*
 * BIDSCloneLayer.cpp
 * can be used to map BIDSLayers to larger dimensions
 *
 *  Created on: Jul 24, 2012
 *      Author: bnowers
 */

#include "HyPerLayer.hpp"
#include "BIDSCloneLayer.hpp"
#include <stdio.h>

#include "../include/default_params.h"

// CloneLayer can be used to implement Sigmoid junctions
namespace PV {
BIDSCloneLayer::BIDSCloneLayer() {
   initialize_base();
}

BIDSCloneLayer::BIDSCloneLayer(const char * name, HyPerCol * hc, const char * origLayerName) {
   initialize_base();
   initialize(name, hc, origLayerName);
}

BIDSCloneLayer::~BIDSCloneLayer()
{
    clayer->V = NULL;
}

int BIDSCloneLayer::initialize_base() {
   sourceLayer = NULL;
   return PV_SUCCESS;
}

int BIDSCloneLayer::initialize(const char * name, HyPerCol * hc, const char * origLayerName) {
   int status_init = HyPerLayer::initialize(name, hc, MAX_CHANNELS);

   V0 = parent->parameters()->value(name, "Vrest", V_REST);
   Vth = parent->parameters()->value(name,"VthRest",VTH_REST);
   InverseFlag = parent->parameters()->value(name,"InverseFlag",INVERSEFLAG);
   SigmoidFlag = parent->parameters()->value(name,"SigmoidFlag",SIGMOIDFLAG);
   SigmoidAlpha = parent->parameters()->value(name,"SigmoidAlpha",SIGMOIDALPHA);


   //if(InverseFlag)   fprintf(stdout,"SigmoidLayer: Inverse flag is set.\n");
   //if(SigmoidFlag)   fprintf(stdout,"SigmoidLayer: True Sigmoid flag is set.\n");

   this->writeSparseActivity = true;

   if (origLayerName==NULL) {
      fprintf(stderr, "SigmoidLayer \"%s\": originalLayerName must be set.\n", name);
      return(EXIT_FAILURE);
   }
   HyPerLayer * origHyPerLayer = parent->getLayerFromName(origLayerName);
   if (origHyPerLayer==NULL) {
      fprintf(stderr, "SigmoidLayer \"%s\" error: originalLayerName \"%s\" is not a layer in the HyPerCol.\n", name, origLayerName);
      return(EXIT_FAILURE);
   }
   sourceLayer = dynamic_cast<LIF *>(origHyPerLayer);
   if (origHyPerLayer==NULL) {
      fprintf(stderr, "SigmoidLayer \"%s\" error: originalLayerName \"%s\" is not a LIF or LIF-derived layer in the HyPerCol.\n", name, origLayerName);
      return(EXIT_FAILURE);
   }

   //free(clayer->V);
   //clayer->V = sourceLayer->getV();

   // don't need conductance channels
   freeChannels();
   sourceLayerA = sourceLayer->getCLayer()->activeIndices;
   sourceLayerNumIndices = &(sourceLayer->getCLayer()->numActive);
   //TODO Check if this works with Pete

   const char * jitterSourceName = parent->parameters()->stringValue(name, "jitterSource");
   BIDSMovieCloneMap *blayer = dynamic_cast<BIDSMovieCloneMap*> (sourceLayer->getParent()->getLayerFromName(jitterSourceName));
   assert(blayer != NULL);
   coords = blayer->getCoords();
   numNodes = blayer->getNumNodes();

   for(int i = 0; i < getNumExtended(); i++){
      this->clayer->activity->data[i] = 0;
   }

   return status_init;
}

int BIDSCloneLayer::mapCoords(){
   for(int i = 0; i < numNodes; i++){
      int index = kIndex(coords[i].xCoord, coords[i].yCoord, 0, clayer->loc.nx, clayer->loc.ny, clayer->loc.nf);
      int indexEx = kIndexExtended(index, clayer->loc.nx, clayer->loc.ny, clayer->loc.nf, clayer->loc.nb);
      this->clayer->activity->data[indexEx] = 0;
   }

   for(unsigned int i = 0; i < *sourceLayerNumIndices; i++){
      int index = kIndex(coords[sourceLayerA[i]].xCoord, coords[sourceLayerA[i]].yCoord, 0, clayer->loc.nx, clayer->loc.ny, clayer->loc.nf);
      int indexEx = kIndexExtended(index, clayer->loc.nx, clayer->loc.ny, clayer->loc.nf, clayer->loc.nb);
      //printf("Coords: %d,%d\tIndex: %d\n", coords[sourceLayerA[i]].xCoord, coords[sourceLayerA[i]].yCoord, index);
      this->clayer->activity->data[indexEx] += 1;
   }
   return PV_SUCCESS;
}

int BIDSCloneLayer::setActivity() {
   pvdata_t * activity = clayer->activity->data;
   memset(activity, 0, sizeof(pvdata_t) * clayer->numExtended);
   return 0;
}

// outputState removed since it was identical to HyPerLayer's outputState

int BIDSCloneLayer::updateState(double timef, double dt) {
   int status = PV_SUCCESS;
   mapCoords();
   if( status == PV_SUCCESS ) status = updateActiveIndices();
   return status;
}

} // end namespace PV

