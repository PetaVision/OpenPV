/*
 * SigmoidLayer.cpp
 *
 *  Created on: May 11, 2011
 *      Author: garkenyon
 */

#include "SigmoidLayer.hpp"
#include <stdio.h>

#include "../include/default_params.h"

// SigmoidLayer can be used to implement Sigmoid junctions
namespace PV {
SigmoidLayer::SigmoidLayer() {
   initialize_base();
}

SigmoidLayer::SigmoidLayer(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

SigmoidLayer::~SigmoidLayer()
{
   // Handled by CloneVLayer destructor
   // clayer->V = NULL;
   // free(sourceLayerName);
}

int SigmoidLayer::initialize_base() {
   // Handled by CloneVLayer
   // sourceLayerName = NULL;
   // sourceLayer = NULL;
   return PV_SUCCESS;
}

int SigmoidLayer::initialize(const char * name, HyPerCol * hc) {
   // if (origLayerName==NULL) {
   //    fprintf(stderr, "SigmoidLayer \"%s\": originalLayerName must be set.\n", name);
   //    exit(EXIT_FAILURE);
   // }
   // sourceLayerName = strdup(origLayerName);
   // if (sourceLayerName == NULL) {
   //    fprintf(stderr, "SigmoidLayer \"%s\" error: rank %d process unable to copy original layer name \"%s\": %s\n", name, parent->columnId(), origLayerName, strerror(errno));
   //    exit(EXIT_FAILURE);
   // }
   int status_init = CloneVLayer::initialize(name, hc);

   if (parent->columnId()==0) {
      if(InverseFlag)   fprintf(stdout,"SigmoidLayer: Inverse flag is set\n");
      if(SigmoidFlag)   fprintf(stdout,"SigmoidLayer: True Sigmoid flag is set\n");
   }

   if (SigmoidAlpha < 0.0f || SigmoidAlpha > 1.0f) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: SigmoidAlpha cannot be negative or greater than 1.\n", parent->parameters()->groupKeywordFromName(name), name);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }

   return status_init;
}

int SigmoidLayer::setParams(PVParams * params) {
   int status = CloneVLayer::setParams(params);
   readVrest(params);
   readVthRest(params);
   readInverseFlag(params);
   readSigmoidFlag(params);
   readSigmoidAlpha(params);
   return status;
}

void SigmoidLayer::readVrest(PVParams * params) {
   V0 = params->value(name, "Vrest", V_REST);
}
void SigmoidLayer::readVthRest(PVParams * params) {
   Vth = params->value(name,"VthRest",VTH_REST);
}
void SigmoidLayer::readInverseFlag(PVParams * params) {
   InverseFlag = params->value(name,"InverseFlag",INVERSEFLAG);
}
void SigmoidLayer::readSigmoidFlag(PVParams * params) {
   SigmoidFlag = params->value(name,"SigmoidFlag",SIGMOIDFLAG);
}
void SigmoidLayer::readSigmoidAlpha(PVParams * params) {
   SigmoidAlpha = params->value(name,"SigmoidAlpha",SIGMOIDALPHA);
}

int SigmoidLayer::communicateInitInfo() {
   int status = CloneVLayer::communicateInitInfo();

   // Handled by CloneVLayer::communicateInitInfo
   // sourceLayer = parent->getLayerFromName(sourceLayerName);
   // if (sourceLayer==NULL) {
   //    if (parent->columnId()==0) {
   //       fprintf(stderr, "SigmoidLayer \"%s\" error: originalLayerName \"%s\" is not a layer in the HyPerCol.\n",
   //               name, sourceLayerName);
   //    }
   //    MPI_Barrier(parent->icCommunicator()->communicator());
   //    exit(EXIT_FAILURE);
   // }
   // const PVLayerLoc * srcLoc = sourceLayer->getLayerLoc();
   // const PVLayerLoc * loc = getLayerLoc();
   // if (srcLoc->nxGlobal != loc->nxGlobal || srcLoc->nyGlobal != loc->nyGlobal || srcLoc->nf != loc->nf) {
   //    if (parent->columnId()==0) {
   //       fprintf(stderr, "SigmoidLayer \"%s\" error: original layer \"%s\" must have the same dimensions.\n", name, sourceLayerName);
   //       fprintf(stderr, "    original: (x=%d, y=%d, f=%d), SigmoidLayer: (x=%d, y=%d, f=%d)\n",
   //               srcLoc->nxGlobal, srcLoc->nyGlobal, srcLoc->nf, loc->nxGlobal, loc->nyGlobal, loc->nf);
   //    }
   //    MPI_Barrier(parent->icCommunicator()->communicator());
   //    exit(EXIT_FAILURE);
   // }

   return status;
}

int SigmoidLayer::allocateDataStructures() {
   int status = CloneVLayer::allocateDataStructures();
   // Handled by CloneVLayer::allocateV
   // free(clayer->V);
   // clayer->V = sourceLayer->getV();

   // Should have been initialized with zero channels, so GSyn should be NULL and freeChannels() call should be unnecessary
   assert(GSyn==NULL);
   // // don't need conductance channels
   // freeChannels();
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

