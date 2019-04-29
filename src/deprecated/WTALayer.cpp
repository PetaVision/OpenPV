/*
 * WTALayer.cpp
 * Author: slundquist
 */

// WTALayer was deprecated on Aug 15, 2018, in favor of WTAConn.

#include "WTALayer.hpp"
#include <assert.h>
#include <fstream>
#include <iostream>
#include <string>

namespace PV {
WTALayer::WTALayer(const char *name, PVParams *params, Communicator const *comm) {
   initialize_base();
   initialize(name, params, comm);
}

WTALayer::~WTALayer() {}

int WTALayer::initialize_base() {
   originalLayerName = NULL;
   originalLayer     = NULL;
   binMax            = 1;
   binMin            = 0;
   return PV_SUCCESS;
}

void WTALayer::initialize(const char *name, PVParams *params, Communicator const *comm) {
   WarnLog() << "WTALayer has been deprecated. Use a WTAConn to a HyPerLayer instead.\n";
   HyPerLayer::initialize(name, params, comm);
}

LayerInputBuffer *WTALayer::createLayerInput() { return nullptr; }

InternalStateBuffer *WTALayer::createInternalState() { return nullptr; }

Response::Status
WTALayer::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = HyPerLayer::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   originalLayer = message->mObjectTable->findObject<HyPerLayer>(originalLayerName);
   FatalIf(
         originalLayer == nullptr,
         "%s: originalLayerName \"%s\" is not a layer in the HyPerCol.\n",
         getDescription_c(),
         originalLayerName);
   if (originalLayer->getInitInfoCommunicatedFlag() == false) {
      return Response::POSTPONE;
   }
   originalLayer->synchronizeMarginWidth(this);
   this->synchronizeMarginWidth(originalLayer);
   const PVLayerLoc *srcLoc = originalLayer->getLayerLoc();
   const PVLayerLoc *loc    = getLayerLoc();
   assert(srcLoc != nullptr && loc != nullptr);
   if (srcLoc->nxGlobal != loc->nxGlobal || srcLoc->nyGlobal != loc->nyGlobal) {
      if (mCommunicator->commRank() == 0) {
         ErrorLog(errorMessage);
         errorMessage.printf(
               "%s: originalLayerName \"%s\" does not have the same dimensions.\n",
               getDescription_c(),
               originalLayerName);
         errorMessage.printf(
               "    original (nx=%d, ny=%d) versus (nx=%d, ny=%d)\n",
               srcLoc->nxGlobal,
               srcLoc->nyGlobal,
               loc->nxGlobal,
               loc->nyGlobal);
      }
      MPI_Barrier(mCommunicator->communicator());
      exit(EXIT_FAILURE);
   }
   if (getLayerLoc()->nf != 1) {
      ErrorLog().printf("%s: WTALayer can only have 1 feature.\n", getDescription_c());
   }
   pvAssert(srcLoc->nx == loc->nx && srcLoc->ny == loc->ny);
   return Response::SUCCESS;
}

void WTALayer::initializeActivity() {}

int WTALayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerLayer::ioParamsFillGroup(ioFlag);
   ioParam_originalLayerName(ioFlag);
   ioParam_binMaxMin(ioFlag);
   return status;
}
void WTALayer::ioParam_originalLayerName(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamStringRequired(ioFlag, name, "originalLayerName", &originalLayerName);
   assert(originalLayerName);
   if (ioFlag == PARAMS_IO_READ && originalLayerName[0] == '\0') {
      if (mCommunicator->commRank() == 0) {
         ErrorLog().printf("%s: originalLayerName must be set.\n", getDescription_c());
      }
      MPI_Barrier(mCommunicator->communicator());
      exit(EXIT_FAILURE);
   }
}

void WTALayer::ioParam_binMaxMin(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "binMax", &binMax, binMax);
   parameters()->ioParamValue(ioFlag, name, "binMin", &binMin, binMin);
   if (ioFlag == PARAMS_IO_READ && binMax <= binMin) {
      if (mCommunicator->commRank() == 0) {
         ErrorLog().printf(
               "%s: binMax (%f) must be greater than binMin (%f).\n",
               getDescription_c(),
               (double)binMax,
               (double)binMin);
      }
      MPI_Barrier(mCommunicator->communicator());
      exit(EXIT_FAILURE);
   }
}

Response::Status WTALayer::updateState(double timef, double dt) {
   float *currA = getActivity();
   float *srcA  = originalLayer->getActivity();

   const PVLayerLoc *loc    = getLayerLoc();
   const PVLayerLoc *srcLoc = originalLayer->getLayerLoc();

   // NF must be one
   assert(loc->nf == 1);
   float stepSize = (float)(binMax - binMin) / (float)srcLoc->nf;

   for (int b = 0; b < srcLoc->nbatch; b++) {
      float *currABatch = currA + b * getNumExtended();
      float *srcABatch  = srcA + b * originalLayer->getNumExtended();
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
