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

int CloneVLayer::initialize_base() { return PV_SUCCESS; }

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

LayerInputBuffer *CloneVLayer::createLayerInput() { return nullptr; }

InternalStateBuffer *CloneVLayer::createInternalState() {
   return nullptr;
   // CloneVLayer will set InternalState to the original layer's InternalState
   // during CommunicateInitInfo
}

Response::Status
CloneVLayer::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = HyPerLayer::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }

   setOriginalLayer();
   pvAssert(mOriginalLayer);
   if (!mOriginalLayer->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }
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
   pvAssert(srcLoc->nx == loc->nx && srcLoc->ny == loc->ny);

   mInternalState = mOriginalLayer->getComponentByType<InternalStateBuffer>();

   return Response::SUCCESS;
}

void CloneVLayer::setOriginalLayer() {
   auto *originalLayerNameParam = getComponentByType<OriginalLayerNameParam>();
   pvAssert(originalLayerNameParam);

   ComponentBasedObject *originalObject = nullptr;
   try {
      originalObject = originalLayerNameParam->findLinkedObject(mObserverTable);
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
   auto *internalState = mInternalState;
   mInternalState      = nullptr;
   auto status         = HyPerLayer::allocateDataStructures();
   mInternalState      = internalState;
   return status;
}

Response::Status
CloneVLayer::registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   InternalStateBuffer *internalState = mInternalState;
   mInternalState                     = nullptr;
   auto status                        = HyPerLayer::registerData(message);
   mInternalState                     = internalState;
   return status;
}

Response::Status CloneVLayer::updateState(double timed, double dt) {
   const PVLayerLoc *loc = getLayerLoc();
   float *A              = mActivity->getActivity();
   float *V              = getV();
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

CloneVLayer::~CloneVLayer() { mInternalState = nullptr; }

} /* namespace PV */
