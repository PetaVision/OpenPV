/*
 * CloneVLayer.cpp
 *
 *  Created on: Aug 15, 2013
 *      Author: pschultz
 */

#include "CloneVLayer.hpp"

namespace PV {

CloneVLayer::CloneVLayer(const char *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

CloneVLayer::CloneVLayer() {
   initialize_base();
   // initialize() gets called by subclass's initialize method
}

int CloneVLayer::initialize_base() {
   numChannels       = 0;
   originalLayerName = NULL;
   originalLayer     = NULL;
   return PV_SUCCESS;
}

int CloneVLayer::initialize(const char *name, HyPerCol *hc) {
   int status = HyPerLayer::initialize(name, hc);
   return status;
}

int CloneVLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerLayer::ioParamsFillGroup(ioFlag);
   ioParam_originalLayerName(ioFlag);
   return status;
}

void CloneVLayer::ioParam_originalLayerName(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamStringRequired(
         ioFlag, name, "originalLayerName", &originalLayerName);
}

void CloneVLayer::ioParam_InitVType(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryParameter(name, "InitVType");
   }
}

Response::Status
CloneVLayer::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = HyPerLayer::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   originalLayer = message->lookup<HyPerLayer>(std::string(originalLayerName));
   if (originalLayer == NULL) {
      if (parent->columnId() == 0) {
         ErrorLog().printf(
               "%s: originalLayerName \"%s\" is not a layer in the HyPerCol.\n",
               getDescription_c(),
               originalLayerName);
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   const PVLayerLoc *srcLoc = originalLayer->getLayerLoc();
   const PVLayerLoc *loc    = getLayerLoc();
   assert(srcLoc != NULL && loc != NULL);
   if (srcLoc->nxGlobal != loc->nxGlobal || srcLoc->nyGlobal != loc->nyGlobal
       || srcLoc->nf != loc->nf) {
      if (parent->columnId() == 0) {
         ErrorLog(errorMessage);
         errorMessage.printf(
               "%s: originalLayerName \"%s\" does not have the same dimensions.\n",
               getDescription_c(),
               originalLayerName);
         errorMessage.printf(
               "    original (nx=%d, ny=%d, nf=%d) versus (nx=%d, ny=%d, nf=%d)\n",
               srcLoc->nxGlobal,
               srcLoc->nyGlobal,
               srcLoc->nf,
               loc->nxGlobal,
               loc->nyGlobal,
               loc->nf);
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   assert(srcLoc->nx == loc->nx && srcLoc->ny == loc->ny);
   return Response::SUCCESS;
}

int CloneVLayer::requireMarginWidth(int marginWidthNeeded, int *marginWidthResult, char axis) {
   HyPerLayer::requireMarginWidth(marginWidthNeeded, marginWidthResult, axis);
   assert(*marginWidthResult >= marginWidthNeeded);
   return PV_SUCCESS;
}

Response::Status CloneVLayer::allocateDataStructures() {
   assert(originalLayer);
   auto status = Response::SUCCESS;
   // Make sure originalLayer has allocated its V buffer before copying its address to clone's V
   // buffer
   if (originalLayer->getDataStructuresAllocatedFlag()) {
      status = HyPerLayer::allocateDataStructures();
   }
   else {
      status = Response::POSTPONE;
   }
   return status;
}

void CloneVLayer::allocateV() {
   assert(originalLayer && originalLayer->getCLayer());
   clayer->V = originalLayer->getV();
   if (getV() == NULL) {
      Fatal().printf(
            "%s: originalLayer \"%s\" has a null V buffer in rank %d process.\n",
            getDescription_c(),
            originalLayerName,
            parent->columnId());
   }
}

int CloneVLayer::requireChannel(int channelNeeded, int *numChannelsResult) {
   if (parent->columnId() == 0) {
      ErrorLog().printf(
            "%s: layers derived from CloneVLayer do not have GSyn channels (requireChannel called "
            "with channel %d)\n",
            getDescription_c(),
            channelNeeded);
   }
   return PV_FAILURE;
}

void CloneVLayer::allocateGSyn() { pvAssert(GSyn == nullptr); }

void CloneVLayer::initializeV() {}

void CloneVLayer::readVFromCheckpoint(Checkpointer *checkpointer) {
   // If we just inherit HyPerLayer::readVFromCheckpoint, we checkpoint V since it is non-null.
   // This is redundant since V is a clone.
}

Response::Status CloneVLayer::registerData(Checkpointer *checkpointer) {
   float *V    = clayer->V;
   clayer->V   = nullptr;
   auto status = HyPerLayer::registerData(checkpointer);
   clayer->V   = V;
   return status;
}

Response::Status CloneVLayer::updateState(double timed, double dt) {
   const PVLayerLoc *loc = getLayerLoc();
   float *A              = clayer->activity->data;
   float *V              = getV();
   int num_channels      = getNumChannels();
   float *gSynHead       = GSyn == NULL ? NULL : GSyn[0];
   int nx                = loc->nx;
   int ny                = loc->ny;
   int nf                = loc->nf;
   int num_neurons       = nx * ny * nf;
   int nbatch            = loc->nbatch;
   setActivity_HyPerLayer(
         nbatch,
         num_neurons,
         A,
         V,
         nx,
         ny,
         loc->nf,
         loc->halo.lt,
         loc->halo.rt,
         loc->halo.dn,
         loc->halo.up);
   return Response::SUCCESS;
}

CloneVLayer::~CloneVLayer() {
   free(originalLayerName);
   clayer->V = NULL;
}

} /* namespace PV */
