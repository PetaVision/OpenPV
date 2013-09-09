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
   originalLayerName = NULL;
   originalLayer = NULL;
   return PV_SUCCESS;
}

int CloneVLayer::initialize(const char * name, HyPerCol * hc) {
   int status = HyPerLayer::initialize(name, hc, 0);
   return status;
}

int CloneVLayer::setParams(PVParams * params) {
   int status = HyPerLayer::setParams(params);
   readOriginalLayerName(params);
   return status;
}

void CloneVLayer::readOriginalLayerName(PVParams * params) {
   const char * original_layer_name = params->stringValue(name, "originalLayerName");
   if (original_layer_name==NULL || original_layer_name[0]=='\0') {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: originalLayerName must be set.\n",
                 parent->parameters()->groupKeywordFromName(name), name);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   originalLayerName = strdup(original_layer_name);
   if (originalLayerName==NULL) {
      fprintf(stderr, "%s \"%s\" error: rank %d process unable to copy originalLayerName \"%s\": %s\n",
              parent->parameters()->groupKeywordFromName(name), name, parent->columnId(), original_layer_name, strerror(errno));
      exit(EXIT_FAILURE);
   }
}

int CloneVLayer::communicateInitInfo() {
   int status = HyPerLayer::communicateInitInfo();
   originalLayer = parent->getLayerFromName(originalLayerName);
   if (originalLayer==NULL) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: originalLayerName \"%s\" is not a layer in the HyPerCol.\n",
                 parent->parameters()->groupKeywordFromName(name), name, originalLayerName);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   originalLayer->synchronizeMarginWidth(this);
   const PVLayerLoc * srcLoc = originalLayer->getLayerLoc();
   const PVLayerLoc * loc = getLayerLoc();
   assert(srcLoc != NULL && loc != NULL);
   if (srcLoc->nxGlobal != loc->nxGlobal || srcLoc->nyGlobal != loc->nyGlobal || srcLoc->nf != loc->nf) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: originalLayerName \"%s\" does not have the same dimensions.\n",
                 parent->parameters()->groupKeywordFromName(name), name, originalLayerName);
         fprintf(stderr, "    original (nx=%d, ny=%d, nf=%d) versus (nx=%d, ny=%d, nf=%d)\n",
                 srcLoc->nxGlobal, srcLoc->nyGlobal, srcLoc->nf, loc->nxGlobal, loc->nyGlobal, loc->nf);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   assert(srcLoc->nx==loc->nx && srcLoc->ny==loc->ny);
   return status;
}

int CloneVLayer::requireMarginWidth(int marginWidthNeeded, int * marginWidthResult) {
   HyPerLayer::requireMarginWidth(marginWidthNeeded, marginWidthResult);
   assert(*marginWidthResult >= marginWidthNeeded);
   originalLayer->requireMarginWidth(marginWidthNeeded, marginWidthResult);
   assert(*marginWidthResult>=marginWidthNeeded);
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
              parent->parameters()->groupKeywordFromName(name), name, originalLayerName, parent->columnId());
      exit(EXIT_FAILURE);
   }
   return PV_SUCCESS;
}

int CloneVLayer::initializeState() {
   int status = PV_SUCCESS;
   PVParams * params = parent->parameters();
   assert(!params->presentAndNotBeenRead(name, "restart"));
   // readRestart(params);
   if( restartFlag ) {
      double timef;
      status = readState(&timef);
   }
   else {
      // TODO Need to make sure that original layer's initializeState has been called before this layer's V.
      status = setActivity();
      if (status == PV_SUCCESS) status = updateActiveIndices();
   }
   return status;
}
int CloneVLayer::checkpointRead(const char * cpDir, double * timed) {
   // If we just call HyPerLayer, we checkpoint V since it is non-null.  This is redundant since V is a clone.
   // So we temporarily set clayer->V to NULL to fool HyPerLayer::checkpointRead into not reading it.
   // A cleaner way to do this would be to have HyPerLayer::checkpointRead call a virtual method checkpointReadV, which can be overridden if unnecessary.
   pvdata_t * V = clayer->V;
   clayer->V = NULL;
   int status = HyPerLayer::checkpointRead(cpDir, timed);
   clayer->V = V;
   return status;
}

int CloneVLayer::checkpointWrite(const char * cpDir) {
   pvdata_t * V = clayer->V;
   clayer->V = NULL;
   int status = HyPerLayer::checkpointWrite(cpDir);
   clayer->V = V;
   return status;
}

int CloneVLayer::doUpdateState(double timed, double dt, const PVLayerLoc * loc, pvdata_t * A,
         pvdata_t * V, int num_channels, pvdata_t * GSynHead, bool spiking,
         unsigned int * active_indices, unsigned int * num_active) {
   update_timer->start();
   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int nb = loc->nb;
   int num_neurons = nx*ny*nf;
   int status = setActivity_HyPerLayer(num_neurons, A, V, nx, ny, nf, nb);
   update_timer->stop();
   return status;
}

CloneVLayer::~CloneVLayer() {
   free(originalLayerName);
   clayer->V = NULL;
}

} /* namespace PV */
