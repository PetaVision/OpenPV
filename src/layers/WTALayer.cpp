/*
 * WTALayer.cpp
 * Author: slundquist
 */

#include "WTALayer.hpp"
#include <assert.h>
#include <fstream>
#include <iostream>
#include <string>

namespace PV {
WTALayer::WTALayer(const char *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

WTALayer::~WTALayer() {}

int WTALayer::initialize_base() {
   numChannels = 0;
   binMax      = 1;
   binMin      = 0;
   return PV_SUCCESS;
}

void WTALayer::setObserverTable() {
   HyPerLayer::setObserverTable();
   auto *originalLayerNameParam = createOriginalLayerNameParam();
   if (originalLayerNameParam) {
      addUniqueComponent(originalLayerNameParam->getDescription(), originalLayerNameParam);
   }
}

OriginalLayerNameParam *WTALayer::createOriginalLayerNameParam() {
   return new OriginalLayerNameParam(name, parent);
}

Response::Status
WTALayer::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = HyPerLayer::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }

   setOriginalLayer();
   pvAssert(mOriginalLayer);
   if (mOriginalLayer->getInitInfoCommunicatedFlag() == false) {
      return Response::POSTPONE;
   }
   mOriginalLayer->synchronizeMarginWidth(this);
   this->synchronizeMarginWidth(mOriginalLayer);
   const PVLayerLoc *srcLoc = mOriginalLayer->getLayerLoc();
   const PVLayerLoc *loc    = getLayerLoc();
   assert(srcLoc != NULL && loc != NULL);
   if (srcLoc->nxGlobal != loc->nxGlobal || srcLoc->nyGlobal != loc->nyGlobal) {
      if (parent->columnId() == 0) {
         ErrorLog(errorMessage);
         errorMessage.printf(
               "%s: originalLayerName \"%s\" does not have the same dimensions.\n",
               getDescription_c(),
               mOriginalLayer->getName());
         errorMessage.printf(
               "    original (nx=%d, ny=%d) versus (nx=%d, ny=%d)\n",
               srcLoc->nxGlobal,
               srcLoc->nyGlobal,
               loc->nxGlobal,
               loc->nyGlobal);
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   if (getLayerLoc()->nf != 1) {
      ErrorLog().printf("%s: WTALayer can only have 1 feature.\n", getDescription_c());
   }
   pvAssert(srcLoc->nx == loc->nx && srcLoc->ny == loc->ny);
   return Response::SUCCESS;
}

void WTALayer::setOriginalLayer() {
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

void WTALayer::allocateV() {
   // Allocate V does nothing since binning does not need a V layer
   clayer->V = NULL;
}

void WTALayer::initializeV() { assert(getV() == NULL); }

void WTALayer::initializeActivity() {}

int WTALayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerLayer::ioParamsFillGroup(ioFlag);
   ioParam_binMaxMin(ioFlag);
   return status;
}

void WTALayer::ioParam_binMaxMin(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "binMax", &binMax, binMax);
   parameters()->ioParamValue(ioFlag, name, "binMin", &binMin, binMin);
   if (ioFlag == PARAMS_IO_READ && binMax <= binMin) {
      if (parent->columnId() == 0) {
         ErrorLog().printf(
               "%s: binMax (%f) must be greater than binMin (%f).\n",
               getDescription_c(),
               (double)binMax,
               (double)binMin);
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
}

Response::Status WTALayer::updateState(double timef, double dt) {
   float *currA = getCLayer()->activity->data;
   float *srcA  = mOriginalLayer->getCLayer()->activity->data;

   const PVLayerLoc *loc    = getLayerLoc();
   const PVLayerLoc *srcLoc = mOriginalLayer->getLayerLoc();

   // NF must be one
   assert(loc->nf == 1);
   float stepSize = (float)(binMax - binMin) / (float)srcLoc->nf;

   for (int b = 0; b < srcLoc->nbatch; b++) {
      float *currABatch = currA + b * getNumExtended();
      float *srcABatch  = srcA + b * mOriginalLayer->getNumExtended();
      // Loop over x and y of the src layer
      for (int yi = 0; yi < srcLoc->ny; yi++) {
         for (int xi = 0; xi < srcLoc->nx; xi++) {
            // Find maximum output value in nf direction
            float maxInput = -99999999;
            float maxIndex = -99999999;
            for (int fi = 0; fi < srcLoc->nf; fi++) {
               int kExt = kIndex(
                     xi,
                     yi,
                     fi,
                     srcLoc->nx + srcLoc->halo.lt + srcLoc->halo.rt,
                     srcLoc->ny + srcLoc->halo.up + srcLoc->halo.dn,
                     srcLoc->nf);
               if (srcABatch[kExt] > maxInput || fi == 0) {
                  maxInput = srcABatch[kExt];
                  maxIndex = fi;
               }
            }
            // Found max index, set output to value
            float outVal = (maxIndex * stepSize) + binMin;
            int kExtOut  = kIndex(
                  xi,
                  yi,
                  0,
                  loc->nx + loc->halo.lt + loc->halo.rt,
                  loc->ny + loc->halo.up + loc->halo.dn,
                  loc->nf);
            currABatch[kExtOut] = outVal;
         }
      }
   }
   return Response::SUCCESS;
}
}
