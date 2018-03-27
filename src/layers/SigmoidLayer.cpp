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
SigmoidLayer::SigmoidLayer() { initialize_base(); }

SigmoidLayer::SigmoidLayer(const char *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

SigmoidLayer::~SigmoidLayer() {}

int SigmoidLayer::initialize_base() { return PV_SUCCESS; }

int SigmoidLayer::initialize(const char *name, HyPerCol *hc) {
   int status_init = CloneVLayer::initialize(name, hc);

   if (parent->columnId() == 0) {
      if (InverseFlag)
         InfoLog().printf("SigmoidLayer: Inverse flag is set\n");
      if (SigmoidFlag)
         InfoLog().printf("SigmoidLayer: True Sigmoid flag is set\n");
   }

   if (SigmoidAlpha < 0.0f || SigmoidAlpha > 1.0f) {
      if (parent->columnId() == 0) {
         ErrorLog().printf(
               "%s: SigmoidAlpha cannot be negative or greater than 1.\n", getDescription_c());
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
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
   parent->parameters()->ioParamValue(ioFlag, name, "Vrest", &V0, (float)V_REST);
}
void SigmoidLayer::ioParam_VthRest(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "VthRest", &Vth, (float)VTH_REST);
}
void SigmoidLayer::ioParam_InverseFlag(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "InverseFlag", &InverseFlag, (bool)INVERSEFLAG);
}
void SigmoidLayer::ioParam_SigmoidFlag(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "SigmoidFlag", &SigmoidFlag, (bool)SIGMOIDFLAG);
}
void SigmoidLayer::ioParam_SigmoidAlpha(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "SigmoidAlpha", &SigmoidAlpha, (float)SIGMOIDALPHA);
}

Response::Status
SigmoidLayer::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   return CloneVLayer::communicateInitInfo(message);
}

Response::Status SigmoidLayer::allocateDataStructures() {
   auto status = CloneVLayer::allocateDataStructures();
   // Should have been initialized with zero channels, so GSyn should be NULL and freeChannels()
   // call should be unnecessary
   pvAssert(GSyn == nullptr);
   return status;
}

int SigmoidLayer::setActivity() {
   float *activity = clayer->activity->data;
   memset(activity, 0, sizeof(float) * clayer->numExtendedAllBatches);
   return 0;
}

Response::Status SigmoidLayer::updateState(double timef, double dt) {
   int status;
   updateState(
         timef,
         dt,
         getLayerLoc(),
         getCLayer()->activity->data,
         getV(),
         0,
         NULL,
         Vth,
         V0,
         SigmoidAlpha,
         SigmoidFlag,
         InverseFlag);
   return Response::SUCCESS;
}

void SigmoidLayer::updateState(
      double timef,
      double dt,
      const PVLayerLoc *loc,
      float *A,
      float *V,
      int num_channels,
      float *gSynHead,
      float Vth,
      float V0,
      float sigmoid_alpha,
      bool sigmoid_flag,
      bool inverse_flag) {
   int nx          = loc->nx;
   int ny          = loc->ny;
   int nf          = loc->nf;
   int num_neurons = nx * ny * nf;
   int nbatch      = loc->nbatch;
   updateV_SigmoidLayer(); // Does nothing as sourceLayer is responsible for updating V.
   setActivity_SigmoidLayer(
         nbatch,
         num_neurons,
         A,
         V,
         nx,
         ny,
         nf,
         loc->halo.lt,
         loc->halo.rt,
         loc->halo.dn,
         loc->halo.up,
         Vth,
         V0,
         sigmoid_alpha,
         sigmoid_flag,
         inverse_flag,
         dt);
}

} // end namespace PV
