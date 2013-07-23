/*
 * SigmoidLayer.cpp
 *
 *  Created on: May 11, 2011
 *      Author: garkenyon
 */

#include "HyPerLayer.hpp"
#include "SigmoidLayer.hpp"
#include <stdio.h>

#include "../include/default_params.h"

// CloneLayer can be used to implement Sigmoid junctions
namespace PV {
SigmoidLayer::SigmoidLayer() {
   initialize_base();
}

SigmoidLayer::SigmoidLayer(const char * name, HyPerCol * hc, const char * origLayerName) {
   initialize_base();
   initialize(name, hc, origLayerName);
}

SigmoidLayer::~SigmoidLayer()
{
    clayer->V = NULL;
}

int SigmoidLayer::initialize_base() {
   sourceLayerName = NULL;
   sourceLayer = NULL;
   return PV_SUCCESS;
}

int SigmoidLayer::initialize(const char * name, HyPerCol * hc, const char * origLayerName) {
   int status_init = HyPerLayer::initialize(name, hc, MAX_CHANNELS);

   V0 = parent->parameters()->value(name, "Vrest", V_REST);
   Vth = parent->parameters()->value(name,"VthRest",VTH_REST);
   InverseFlag = parent->parameters()->value(name,"InverseFlag",INVERSEFLAG);
   SigmoidFlag = parent->parameters()->value(name,"SigmoidFlag",SIGMOIDFLAG);
   SigmoidAlpha = parent->parameters()->value(name,"SigmoidAlpha",SIGMOIDALPHA);

   if (parent->columnId()==0) {
      if(InverseFlag)   fprintf(stdout,"SigmoidLayer: Inverse flag is set\n");
      if(SigmoidFlag)   fprintf(stdout,"SigmoidLayer: True Sigmoid flag is set\n");
   }

   if (origLayerName==NULL) {
      fprintf(stderr, "SigmoidLayer \"%s\": originalLayerName must be set.\n", name);
      return(EXIT_FAILURE);
   }
   sourceLayerName = strdup(origLayerName);
   if (sourceLayerName==NULL) {
      fprintf(stderr, "SigmoidLayer \"%s\" error: unable to copy originalLayerName \"%s\": %s\n", name, origLayerName, strerror(errno));
      exit(EXIT_FAILURE);
   }

   // Moved to communicateInitInfo()
   // HyPerLayer * origHyPerLayer = parent->getLayerFromName(origLayerName);
   // if (origHyPerLayer==NULL) {
   //    fprintf(stderr, "SigmoidLayer \"%s\" error: originalLayerName \"%s\" is not a layer in the HyPerCol.\n", name, origLayerName);
   //    return(EXIT_FAILURE);
   // }
   // sourceLayer = dynamic_cast<LIF *>(origHyPerLayer);
   // if (origHyPerLayer==NULL) {
   //    fprintf(stderr, "SigmoidLayer \"%s\" error: originalLayerName \"%s\" is not a LIF or LIF-derived layer in the HyPerCol.\n", name, origLayerName);
   //    return(EXIT_FAILURE);
   // }

   // Moved to allocateInitInfo()
   // free(clayer->V);
   // clayer->V = sourceLayer->getV();
   //
   // // don't need conductance channels
   // freeChannels();

   return status_init;
}

int SigmoidLayer::communicateInitInfo() {
   int status = HyPerLayer::communicateInitInfo();

   HyPerLayer * origHyPerLayer = parent->getLayerFromName(sourceLayerName);
   if (origHyPerLayer==NULL) {
      fprintf(stderr, "SigmoidLayer \"%s\" error: originalLayerName \"%s\" is not a layer in the HyPerCol.\n", name, sourceLayerName);
      return(EXIT_FAILURE);
   }
   sourceLayer = dynamic_cast<LIF *>(origHyPerLayer);
   if (origHyPerLayer==NULL) {
      fprintf(stderr, "SigmoidLayer \"%s\" error: originalLayerName \"%s\" is not a LIF or LIF-derived layer in the HyPerCol.\n", name, sourceLayerName);
      return(EXIT_FAILURE);
   }

   return status;
}

int SigmoidLayer::allocateDataStructures() {
   int status = HyPerLayer::allocateDataStructures();
   free(clayer->V);
   clayer->V = sourceLayer->getV();

   // don't need conductance channels
   freeChannels();
   return status;
}


int SigmoidLayer::setActivity() {
   pvdata_t * activity = clayer->activity->data;
   memset(activity, 0, sizeof(pvdata_t) * clayer->numExtended);
   return 0;
}

int SigmoidLayer::updateState(double timef, double dt) {
   int status;
   status = updateState(timef, dt, getLayerLoc(), getCLayer()->activity->data, getV(), 0, NULL, Vth, V0, SigmoidAlpha, SigmoidFlag, InverseFlag, getCLayer()->activeIndices, &getCLayer()->numActive);
   if( status == PV_SUCCESS ) status = updateActiveIndices();
   return status;
}

int SigmoidLayer::updateState(double timef, double dt, const PVLayerLoc * loc, pvdata_t * A, pvdata_t * V,  int num_channels, pvdata_t * gSynHead, float Vth, float V0, float sigmoid_alpha, bool sigmoid_flag, bool inverse_flag, unsigned int * active_indices, unsigned int * num_active) {
   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int num_neurons = nx*ny*nf;
   updateV_SigmoidLayer(); // Does nothing as sourceLayer is responsible for updating V.
   setActivity_SigmoidLayer(num_neurons, A, V, nx, ny, nf, loc->nb, Vth, V0, sigmoid_alpha, sigmoid_flag, inverse_flag, dt);
   // resetGSynBuffers(); // Since sourceLayer updates V, this->GSyn is not used
   return PV_SUCCESS;
}



} // end namespace PV

