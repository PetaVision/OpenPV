/*
 * BinningTestProbe.cpp
 *
 *  Created on: Jan 15, 2015
 *      Author: slundquist
 */

#include "BinningTestProbe.hpp"
#include <components/BinningActivityBuffer.hpp>

namespace PV {

BinningTestProbe::BinningTestProbe(const char *name, PVParams *params, Communicator *comm) {
   LayerProbe::initialize(name, params, comm);
}

Response::Status
BinningTestProbe::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = LayerProbe::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   mBinningLayer = dynamic_cast<BinningLayer *>(getTargetLayer());
   FatalIf(
         mBinningLayer == nullptr,
         "%s requires the target layer to be a BinningLayer.\n",
         getDescription_c());
   return Response::SUCCESS;
}

Response::Status BinningTestProbe::outputState(double simTime, double deltaTime) {
   if (simTime == 0.0) {
      return Response::SUCCESS;
   }
   // Grab layer size
   const PVLayerLoc *loc = mBinningLayer->getLayerLoc();
   int nx                = loc->nx;
   int ny                = loc->ny;
   int nf                = loc->nf;
   int kx0               = loc->kx0;
   int ky0               = loc->ky0;
   int nxGlobal          = loc->nxGlobal;
   int nyGlobal          = loc->nyGlobal;
   int nxExt             = nx + loc->halo.lt + loc->halo.rt;
   int nyExt             = ny + loc->halo.lt + loc->halo.rt;
   int nxGlobalExt       = nxGlobal + loc->halo.lt + loc->halo.rt;
   int nyGlobalExt       = nyGlobal + loc->halo.lt + loc->halo.rt;
   // Grab the activity layer of current layer
   const float *A = mBinningLayer->getLayerData();

   // Grab BinSigma from BinningLayer, which is contained in a component.
   auto *activityComponent = mBinningLayer->getComponentByType<ComponentBasedObject>();
   pvAssert(activityComponent);
   auto *binningActivityBuffer = activityComponent->getComponentByType<BinningActivityBuffer>();
   pvAssert(binningActivityBuffer);
   const float binSigma = binningActivityBuffer->getBinSigma();

   // We only care about restricted space
   for (int iY = loc->halo.up; iY < ny + loc->halo.up; iY++) {
      for (int iX = loc->halo.up; iX < nx + loc->halo.lt; iX++) {
         for (int iF = 0; iF < nf; iF++) {
            int origIndexGlobal    = kIndex(iX + kx0, iY + ky0, 0, nxGlobalExt, nyGlobalExt, 1);
            int binningIndexGlobal = kIndex(iX + kx0, iY + ky0, iF, nxGlobalExt, nyGlobalExt, nf);
            int binningIndexLocal  = kIndex(iX, iY, iF, nxExt, nyExt, nf);
            float observedValue    = A[binningIndexLocal];
            if (binSigma == 0) {
               // Based on the input image, F index should be floor(origIndex/255*32), except
               // that if origIndex==255, F index should be 31.
               float binnedIndex =
                     std::floor((float)origIndexGlobal / 255.0f * 32.0f) - (origIndexGlobal == 255);
               float correctValue = iF == binnedIndex;
               FatalIf(
                     observedValue != correctValue,
                     "%s, extended global location x=%d, y=%d, f=%d, expected %f, observed %f.\n",
                     getTargetLayer()->getDescription_c(),
                     iX + kx0,
                     iY + ky0,
                     iF,
                     (double)correctValue,
                     (double)observedValue);
            }
            else {
               // Map feature index to the center of its bin
               float binCenter = ((float)iF + 0.5f) / nf; // Assumes maxBin is 1 and minBin is zero
               // Determine number of bins away the input value is from the bin center
               float inputValue   = (float)origIndexGlobal / 255.0f;
               float binOffset    = (binCenter - inputValue) * (float)loc->nf;
               float correctValue = exp(-binOffset * binOffset / (2 * binSigma * binSigma));
               FatalIf(
                     std::fabs(observedValue - correctValue) > 0.0001f,
                     "%s, extended global location x=%d, y=%d, f=%d, expected %f, observed %f.\n",
                     getTargetLayer()->getDescription_c(),
                     iX + kx0,
                     iY + ky0,
                     iF,
                     (double)correctValue,
                     (double)observedValue);
            }
         }
      }
   }
   return Response::SUCCESS;
}

} // end namespace PV
