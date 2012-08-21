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

BIDSCloneLayer::BIDSCloneLayer(const char * name, HyPerCol * hc, LIF * originalLayer) {
   initialize_base();
   initialize(name, hc, originalLayer);
}

BIDSCloneLayer::~BIDSCloneLayer()
{
    clayer->V = NULL;
}

int BIDSCloneLayer::initialize_base() {
   sourceLayer = NULL;
   return PV_SUCCESS;
}

int BIDSCloneLayer::initialize(const char * name, HyPerCol * hc, LIF * clone) {
   int status_init = HyPerLayer::initialize(name, hc, MAX_CHANNELS);

   V0 = parent->parameters()->value(name, "Vrest", V_REST);
   Vth = parent->parameters()->value(name,"VthRest",VTH_REST);
   InverseFlag = parent->parameters()->value(name,"InverseFlag",INVERSEFLAG);
   SigmoidFlag = parent->parameters()->value(name,"SigmoidFlag",SIGMOIDFLAG);
   SigmoidAlpha = parent->parameters()->value(name,"SigmoidAlpha",SIGMOIDALPHA);


   if(InverseFlag)   fprintf(stdout,"SigmoidLayer: Inverse flag is set");
   if(SigmoidFlag)   fprintf(stdout,"SigmoidLayer: True Sigmoid flag is set");

   this->spikingFlag = true;
   sourceLayer = clone;
   //free(clayer->V);
   //clayer->V = sourceLayer->getV();

   // don't need conductance channels
   freeChannels();
   sourceLayerA = sourceLayer->getCLayer()->activeIndices;
   sourceLayerNumIndices = &(sourceLayer->getCLayer()->numActive);
   BIDSLayer *blayer = dynamic_cast<BIDSLayer*> (sourceLayer);
   coords = blayer->getCoords();
   numNodes = blayer->numNodes;

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

int BIDSCloneLayer::outputState(float timef, bool last){
   int status = PV_SUCCESS;

   for (int i = 0; i < numProbes; i++) {
      probes[i]->outputState(timef);
   }

   if (timef >= writeTime && writeStep >= 0) {
      writeTime += writeStep;
      if (spikingFlag != 0) {
         status = writeActivitySparse(timef);
      }
      else {
         if (writeNonspikingActivity) {
            status = writeActivity(timef);
         }
      }
   }
   return status;
}

int BIDSCloneLayer::updateState(float timef, float dt) {
   int status = PV_SUCCESS;
   mapCoords();
   if( status == PV_SUCCESS ) status = updateActiveIndices();
   return status;
}

} // end namespace PV

