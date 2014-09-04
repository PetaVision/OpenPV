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

BIDSCloneLayer::BIDSCloneLayer(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

BIDSCloneLayer::~BIDSCloneLayer()
{
   // Handled by CloneVLayer destructor
   // clayer->V = NULL;
    free(jitterSourceName);
}

int BIDSCloneLayer::initialize_base() {
   return PV_SUCCESS;
}

int BIDSCloneLayer::initialize(const char * name, HyPerCol * hc) {
   int status_init = CloneVLayer::initialize(name, hc);

   return status_init;
}

int BIDSCloneLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = CloneVLayer::ioParamsFillGroup(ioFlag);
   ioParam_jitterSource(ioFlag);
   return status;
}

void BIDSCloneLayer::ioParam_writeSparseActivity(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      writeSparseActivity = true;
      parent->parameters()->handleUnnecessaryParameter(name, "writeSparseActivity", writeSparseActivity);
   }
}

void BIDSCloneLayer::ioParam_writeSparseValues(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      writeSparseValues = false;
      parent->parameters()->handleUnnecessaryParameter(name, "writeSparseValues", writeSparseValues);
   }
}

void BIDSCloneLayer::ioParam_jitterSource(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "jitterSource", &jitterSourceName);
}

int BIDSCloneLayer::communicateInitInfo() {
   int status = CloneVLayer::communicateInitInfo();

   return status;
}

int BIDSCloneLayer::allocateDataStructures() {
   int status = CloneVLayer::allocateDataStructures();

   assert(GSyn==NULL);

   BIDSMovieCloneMap *blayer = dynamic_cast<BIDSMovieCloneMap*> (originalLayer->getParent()->getLayerFromName(jitterSourceName));
   if (blayer==NULL) {
      fprintf(stderr, "BIDSCloneLayer \"%s\": jitterSource \"%s\" must be a BIDSMovieCloneMap.\n", name, jitterSourceName);
      abort();
   }
   coords = blayer->getCoords();
   numNodes = blayer->getNumNodes();

   for(int i = 0; i < getNumExtended(); i++){
      this->clayer->activity->data[i] = 0;
   }
   return status;
}

int BIDSCloneLayer::mapCoords(){
   for(int i = 0; i < numNodes; i++){
      int index = kIndex(coords[i].xCoord, coords[i].yCoord, 0, clayer->loc.nx, clayer->loc.ny, clayer->loc.nf);
      int indexEx = kIndexExtended(index, clayer->loc.nx, clayer->loc.ny, clayer->loc.nf, clayer->loc.halo.lt, clayer->loc.halo.rt, clayer->loc.halo.dn, clayer->loc.halo.up);
      this->clayer->activity->data[indexEx] = 0;
   }

   unsigned int * sourceLayerA = getSourceActiveIndices();
   unsigned int sourceLayerNumIndices = getSourceNumActive();
   for(unsigned int i = 0; i < sourceLayerNumIndices; i++){
      int index = kIndex(coords[sourceLayerA[i]].xCoord, coords[sourceLayerA[i]].yCoord, 0, clayer->loc.nx, clayer->loc.ny, clayer->loc.nf);
      int indexEx = kIndexExtended(index, clayer->loc.nx, clayer->loc.ny, clayer->loc.nf, clayer->loc.halo.lt, clayer->loc.halo.rt, clayer->loc.halo.dn, clayer->loc.halo.up);
      this->clayer->activity->data[indexEx] += 1;
   }
   return PV_SUCCESS;
}

int BIDSCloneLayer::setActivity() {
   pvdata_t * activity = clayer->activity->data;
   memset(activity, 0, sizeof(pvdata_t) * clayer->numExtended);
   return 0;
}

int BIDSCloneLayer::updateState(double timef, double dt) {
   int status = PV_SUCCESS;
   mapCoords();
   if( status == PV_SUCCESS ) status = updateActiveIndices();
   return status;
}

} // end namespace PV

