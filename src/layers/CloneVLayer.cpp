/*
 * CloneVLayer.cpp
 *
 *  Created on: Aug 15, 2013
 *      Author: pschultz
 */

#include "CloneVLayer.hpp"

namespace PV {

CloneVLayer::CloneVLayer(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

CloneVLayer::CloneVLayer() {
   initialize_base();
   // initialize() gets called by subclass's initialize method
}

int CloneVLayer::initialize_base() {
   numChannels = 0;
   originalLayerName = NULL;
   originalLayer = NULL;
   return PV_SUCCESS;
}

int CloneVLayer::initialize(const char * name, HyPerCol * hc) {
   int status = HyPerLayer::initialize(name, hc);
   return status;
}

int CloneVLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerLayer::ioParamsFillGroup(ioFlag);
   ioParam_originalLayerName(ioFlag);
   return status;
}

void CloneVLayer::ioParam_originalLayerName(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "originalLayerName", &originalLayerName);
}

void CloneVLayer::ioParam_InitVType(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ) {
      parent->parameters()->handleUnnecessaryParameter(name, "InitVType");
   }
}

int CloneVLayer::communicateInitInfo() {
   int status = HyPerLayer::communicateInitInfo();
   originalLayer = parent->getLayerFromName(originalLayerName);
   if (originalLayer==NULL) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: originalLayerName \"%s\" is not a layer in the HyPerCol.\n",
                 getKeyword(), name, originalLayerName);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   //originalLayer->synchronizeMarginWidth(this);
   const PVLayerLoc * srcLoc = originalLayer->getLayerLoc();
   const PVLayerLoc * loc = getLayerLoc();
   assert(srcLoc != NULL && loc != NULL);
   if (srcLoc->nxGlobal != loc->nxGlobal || srcLoc->nyGlobal != loc->nyGlobal || srcLoc->nf != loc->nf) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: originalLayerName \"%s\" does not have the same dimensions.\n",
                 getKeyword(), name, originalLayerName);
         fprintf(stderr, "    original (nx=%d, ny=%d, nf=%d) versus (nx=%d, ny=%d, nf=%d)\n",
                 srcLoc->nxGlobal, srcLoc->nyGlobal, srcLoc->nf, loc->nxGlobal, loc->nyGlobal, loc->nf);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   assert(srcLoc->nx==loc->nx && srcLoc->ny==loc->ny);
   return status;
}

int CloneVLayer::requireMarginWidth(int marginWidthNeeded, int * marginWidthResult, char axis) {
   HyPerLayer::requireMarginWidth(marginWidthNeeded, marginWidthResult, axis);
   assert(*marginWidthResult >= marginWidthNeeded);
   // The code below is handled by the synchronizeMarginWidth call in communicateInitInfo
   // originalLayer->requireMarginWidth(marginWidthNeeded, marginWidthResult, axis);
   // assert(*marginWidthResult>=marginWidthNeeded);
   return PV_SUCCESS;
}

int CloneVLayer::allocateDataStructures() {
   assert(originalLayer);
   int status = PV_SUCCESS;
   // Make sure originalLayer has allocated its V buffer before copying its address to clone's V buffer
   if (originalLayer->getDataStructuresAllocatedFlag()) {
      status = HyPerLayer::allocateDataStructures();
   }
   else {
      status = PV_POSTPONE;
   }
   return status;
}

int CloneVLayer::allocateV() {
   assert(originalLayer && originalLayer->getCLayer());
   clayer->V = originalLayer->getV();
   if (getV()==NULL) {
      fprintf(stderr, "%s \"%s\": originalLayer \"%s\" has a null V buffer in rank %d process.\n",
              getKeyword(), name, originalLayerName, parent->columnId());
      exit(EXIT_FAILURE);
   }
   return PV_SUCCESS;
}

int CloneVLayer::requireChannel(int channelNeeded, int * numChannelsResult) {
   if (parent->columnId()==0) {
      fprintf(stderr, "%s \"%s\": layers derived from CloneVLayer do not have GSyn channels (requireChannel called with channel %d)\n", getKeyword(), name, channelNeeded);
   }
   return PV_FAILURE;
}

int CloneVLayer::allocateGSyn() {
   assert(GSyn == NULL);
   return PV_SUCCESS;
}

int CloneVLayer::initializeV() {
   return PV_SUCCESS;
}

int CloneVLayer::readVFromCheckpoint(const char * cpDir, double * timeptr) {
   // If we just inherit HyPerLayer::readVFromCheckpoint, we checkpoint V since it is non-null.  This is redundant since V is a clone.
   return PV_SUCCESS;
}

int CloneVLayer::checkpointWrite(const char * cpDir) {
   pvdata_t * V = clayer->V;
   clayer->V = NULL;
   int status = HyPerLayer::checkpointWrite(cpDir);
   clayer->V = V;
   return status;
}

int CloneVLayer::doUpdateState(double timed, double dt, const PVLayerLoc * loc, pvdata_t * A,
         pvdata_t * V, int num_channels, pvdata_t * GSynHead){
   //update_timer->start();
   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int num_neurons = nx*ny*nf;
   int nbatch = loc->nbatch;
   int status = setActivity_HyPerLayer(nbatch, num_neurons, A, V, nx, ny, loc->nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
   //update_timer->stop();
   return status;
}

CloneVLayer::~CloneVLayer() {
   free(originalLayerName);
   clayer->V = NULL;
}

BaseObject * createCloneVLayer(char const * name, HyPerCol * hc) {
   return hc ? new CloneVLayer(name, hc) : NULL;
}

} /* namespace PV */
