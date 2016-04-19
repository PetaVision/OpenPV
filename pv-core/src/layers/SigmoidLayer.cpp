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
   int status_init = CloneVLayer::initialize(name, hc);

   if (parent->columnId()==0) {
      if(InverseFlag)   fprintf(stdout,"SigmoidLayer: Inverse flag is set\n");
      if(SigmoidFlag)   fprintf(stdout,"SigmoidLayer: True Sigmoid flag is set\n");
   }

   if (SigmoidAlpha < 0.0f || SigmoidAlpha > 1.0f) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: SigmoidAlpha cannot be negative or greater than 1.\n", getKeyword(), name);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }

   return status_init;
}

int SigmoidLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = CloneVLayer::ioParamsFillGroup(ioFlag);
   ioParam_Vrest(ioFlag);
   ioParam_VthRest(ioFlag);
   ioParam_InverseFlag(ioFlag);
   ioParam_SigmoidFlag(ioFlag);
   ioParam_SigmoidAlpha(ioFlag);
   return status;
}

void SigmoidLayer::ioParam_Vrest(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "Vrest", &V0, (float) V_REST);
}
void SigmoidLayer::ioParam_VthRest(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "VthRest", &Vth, (float) VTH_REST);
}
void SigmoidLayer::ioParam_InverseFlag(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "InverseFlag", &InverseFlag, (bool) INVERSEFLAG);
}
void SigmoidLayer::ioParam_SigmoidFlag(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "SigmoidFlag", &SigmoidFlag, (bool) SIGMOIDFLAG);
}
void SigmoidLayer::ioParam_SigmoidAlpha(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "SigmoidAlpha", &SigmoidAlpha, (float) SIGMOIDALPHA);
}

int SigmoidLayer::communicateInitInfo() {
   int status = CloneVLayer::communicateInitInfo();

   return status;
}

int SigmoidLayer::allocateDataStructures() {
   int status = CloneVLayer::allocateDataStructures();
   // Should have been initialized with zero channels, so GSyn should be NULL and freeChannels() call should be unnecessary
   assert(GSyn==NULL);
   return status;
}


int SigmoidLayer::setActivity() {
   pvdata_t * activity = clayer->activity->data;
   memset(activity, 0, sizeof(pvdata_t) * clayer->numExtendedAllBatches);
   return 0;
}

int SigmoidLayer::updateState(double timef, double dt) {
   int status;
   status = updateState(timef, dt, getLayerLoc(), getCLayer()->activity->data, getV(), 0, NULL, Vth, V0, SigmoidAlpha, SigmoidFlag, InverseFlag);
   return status;
}

int SigmoidLayer::updateState(double timef, double dt, const PVLayerLoc * loc, pvdata_t * A, pvdata_t * V,  int num_channels, pvdata_t * gSynHead, float Vth, float V0, float sigmoid_alpha, bool sigmoid_flag, bool inverse_flag) {
   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int num_neurons = nx*ny*nf;
   int nbatch = loc->nbatch;
   updateV_SigmoidLayer(); // Does nothing as sourceLayer is responsible for updating V.
   setActivity_SigmoidLayer(nbatch, num_neurons, A, V, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up, Vth, V0, sigmoid_alpha, sigmoid_flag, inverse_flag, dt);
   // resetGSynBuffers(); // Since sourceLayer updates V, this->GSyn is not used
   return PV_SUCCESS;
}

BaseObject * createSigmoidLayer(char const * name, HyPerCol * hc) {
   return hc ? new SigmoidLayer(name, hc) : NULL;
}

} // end namespace PV

