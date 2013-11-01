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

   // this->writeSparseActivity = true; // Instead override readWriteSparseActivity

   // Handled by CloneVLayer::initialize()
   // if (origLayerName==NULL) {
   //    fprintf(stderr, "BIDSCloneLayer \"%s\" error: origLayerName must be set.\n", name);
   //    exit(EXIT_FAILURE);
   // }
   // sourceLayerName = strdup(origLayerName);
   // if (sourceLayerName==NULL) {
   //    fprintf(stderr, "BIDSCloneLayer \"%s\" error: unable to copy origLayerName \"%s\": %s.\n", name, origLayerName, strerror(errno));
   //    exit(EXIT_FAILURE);
   // }

   // Moved to readJitterSource()
   // const char * jitter_source_name = parent->parameters()->stringValue(name, "jitterSource");
   // jitterSourceName = strdup(jitter_source_name);

   return status_init;
}

int BIDSCloneLayer::setParams(PVParams * params) {
   int status = CloneVLayer::setParams(params);
   readJitterSource(params);
   return status;
}

void BIDSCloneLayer::readWriteSparseActivity(PVParams * params) {
   this->writeSparseActivity = true;
   handleUnnecessaryBoolParameter("writeSparseActivity", writeSparseActivity);
}

void BIDSCloneLayer::readWriteSparseValues(PVParams * params) {
   this->writeSparseActivity = false;
   handleUnnecessaryBoolParameter("writeSparseValues", writeSparseValues);
}

void BIDSCloneLayer::readJitterSource(PVParams * params) {
   const char * jitter_source_name = params->stringValue(name, "jitterSource");
   jitterSourceName = strdup(jitter_source_name);
}

int BIDSCloneLayer::communicateInitInfo() {
   int status = CloneVLayer::communicateInitInfo();

   // Handled by CloneVLayer::communicateInitInfo()
   // HyPerLayer * origHyPerLayer = parent->getLayerFromName(sourceLayerName);
   // if (origHyPerLayer==NULL) {
   //    fprintf(stderr, "SigmoidLayer \"%s\" error: originalLayerName \"%s\" is not a layer in the HyPerCol.\n", name, sourceLayerName);
   //    return(EXIT_FAILURE);
   // }
   // sourceLayer = dynamic_cast<LIF *>(origHyPerLayer);
   // if (origHyPerLayer==NULL) {
   //    fprintf(stderr, "SigmoidLayer \"%s\" error: originalLayerName \"%s\" is not a LIF or LIF-derived layer in the HyPerCol.\n", name, sourceLayerName);
   //    return(EXIT_FAILURE);
   // }

   return status;
}

int BIDSCloneLayer::allocateDataStructures() {
   int status = CloneVLayer::allocateDataStructures();

   // Handled by CloneVLayer::allocateV()
   // free(clayer->V);
   // clayer->V = sourceLayer->getV();

   // Should have been initialized with zero channels, so GSyn should be NULL and freeChannels() call should be unnecessary
   assert(GSyn==NULL);
   // // don't need conductance channels
   // freeChannels();

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
      int indexEx = kIndexExtended(index, clayer->loc.nx, clayer->loc.ny, clayer->loc.nf, clayer->loc.nb);
      this->clayer->activity->data[indexEx] = 0;
   }

   unsigned int * sourceLayerA = getSourceActiveIndices();
   unsigned int sourceLayerNumIndices = getSourceNumActive();
   for(unsigned int i = 0; i < sourceLayerNumIndices; i++){
      int index = kIndex(coords[sourceLayerA[i]].xCoord, coords[sourceLayerA[i]].yCoord, 0, clayer->loc.nx, clayer->loc.ny, clayer->loc.nf);
      int indexEx = kIndexExtended(index, clayer->loc.nx, clayer->loc.ny, clayer->loc.nf, clayer->loc.nb);
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

