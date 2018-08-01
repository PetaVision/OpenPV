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
   numChannels = 0;
   return PV_SUCCESS;
}

int CloneVLayer::initialize(const char *name, HyPerCol *hc) {
   int status = HyPerLayer::initialize(name, hc);
   return status;
}

void CloneVLayer::setObserverTable() {
   HyPerLayer::setObserverTable();
   auto *originalLayerNameParam = createOriginalLayerNameParam();
   if (originalLayerNameParam) {
      addUniqueComponent(originalLayerNameParam->getDescription(), originalLayerNameParam);
   }
}

OriginalLayerNameParam *CloneVLayer::createOriginalLayerNameParam() {
   return new OriginalLayerNameParam(name, parent);
}

void CloneVLayer::ioParam_InitVType(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      parameters()->handleUnnecessaryParameter(name, "InitVType");
   }
}

Response::Status
CloneVLayer::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = HyPerLayer::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }

   setOriginalLayer();
   pvAssert(mOriginalLayer);
   const PVLayerLoc *srcLoc = mOriginalLayer->getLayerLoc();
   const PVLayerLoc *loc    = getLayerLoc();
   assert(srcLoc != NULL && loc != NULL);
   if (srcLoc->nxGlobal != loc->nxGlobal || srcLoc->nyGlobal != loc->nyGlobal
       || srcLoc->nf != loc->nf) {
      if (parent->getCommunicator()->commRank() == 0) {
         ErrorLog(errorMessage);
         errorMessage.printf(
               "%s: originalLayerName \"%s\" does not have the same dimensions.\n",
               getDescription_c(),
               mOriginalLayer->getName());
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

void CloneVLayer::setOriginalLayer() {
   auto *originalLayerNameParam = getComponentByType<OriginalLayerNameParam>();
   pvAssert(originalLayerNameParam);

   ComponentBasedObject *originalObject = nullptr;
   try {
      originalObject = originalLayerNameParam->findLinkedObject(mObserverTable.getObjectMap());
   } catch (std::invalid_argument &e) {
      Fatal().printf("%s: %s\n", getDescription_c(), e.what());
   }
   pvAssert(originalObject);

   mOriginalLayer = dynamic_cast<HyPerLayer *>(originalObject);
   if (mOriginalLayer == nullptr) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         ErrorLog().printf(
               "%s: originalLayerName \"%s\" is not a layer in the HyPerCol.\n",
               getDescription_c(),
               originalObject->getName());
      }
      MPI_Barrier(parent->getCommunicator()->globalCommunicator());
      exit(EXIT_FAILURE);
   }
}

Response::Status CloneVLayer::allocateDataStructures() {
   assert(mOriginalLayer);
   auto status = Response::SUCCESS;
   // Make sure mOriginalLayer has allocated its V buffer before copying its address to
   // clone's V buffer
   if (mOriginalLayer->getDataStructuresAllocatedFlag()) {
      status = HyPerLayer::allocateDataStructures();
   }
   else {
      status = Response::POSTPONE;
   }
   return status;
}

void CloneVLayer::allocateV() {
   assert(mOriginalLayer && mOriginalLayer->getCLayer());
   clayer->V = mOriginalLayer->getV();
   if (getV() == NULL) {
      Fatal().printf(
            "%s: original layer \"%s\" has a null V buffer in rank %d process.\n",
            getDescription_c(),
            mOriginalLayer->getName(),
            parent->getCommunicator()->globalCommRank());
   }
}

int CloneVLayer::requireChannel(int channelNeeded, int *numChannelsResult) {
   if (parent->getCommunicator()->globalCommRank() == 0) {
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

Response::Status
CloneVLayer::registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   float *V    = clayer->V;
   clayer->V   = nullptr;
   auto status = HyPerLayer::registerData(message);
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

CloneVLayer::~CloneVLayer() { clayer->V = NULL; }

} /* namespace PV */
