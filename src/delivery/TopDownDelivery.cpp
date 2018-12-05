/*
 * TopDownDelivery.cpp
 *
 *  Created on: Aug 24, 2017
 *      Author: Pete Schultz
 */

#include "TopDownDelivery.hpp"
#include "columns/HyPerCol.hpp"

namespace PV {

TopDownDelivery::TopDownDelivery(char const *name, HyPerCol *hc) { initialize(name, hc); }

int TopDownDelivery::initialize(char const *name, HyPerCol *hc) {
   return BaseDelivery::initialize(name, hc);
   mDeliverCount = 0;
}

void TopDownDelivery::setObjectType() { mObjectType = "TopDownDelivery"; }

int TopDownDelivery::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = BaseDelivery::ioParamsFillGroup(ioFlag);
   ioParam_displayPeriod(ioFlag);
   ioParam_zeroRatio(ioFlag);
   return status;
}

void TopDownDelivery::ioParam_displayPeriod(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "displayPeriod", &mDisplayPeriod, mDisplayPeriod /*default*/, true /*warn if absent*/);
   FatalIf(mDisplayPeriod <= 0, "TopDownDelivery requires a displayPeriod > 0\n");
}

void TopDownDelivery::ioParam_zeroRatio(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "zeroRatio", &mZeroRatio, mZeroRatio /*default*/, true /*warn if absent*/);
   FatalIf(mZeroRatio <= 0 || mZeroRatio > 0.5, "TopDownDelivery requires a zeroRatio > 0 and <= 0.5\n");
}


void TopDownDelivery::deliver() {
   if (mChannelCode == CHANNEL_NOUPDATE) {
      return;
   }

   float scale = 0.0f;
   float prog  = (float)mDeliverCount / mDisplayPeriod;

   if (prog > mZeroRatio) {
      if (prog >= 1.0f - mZeroRatio) {
         scale = 1.0f;
      }
      else {
         // The 0.5f divide by zero is caught above
         scale = (prog - mZeroRatio) / (1.0f - mZeroRatio * 2.0f);
      }
   }

   // This is probably off by 1 timestep because of the first deliver on
   // network startup, but it shouldn't affect it enough to matter
   mDeliverCount = (mDeliverCount + 1) % mDisplayPeriod;

   int delay                         = mSingleArbor->getDelay(0);
   PVLayerCube const preActivityCube = mPreLayer->getPublisher()->createCube(delay);
   PVLayerLoc const &preLoc          = preActivityCube.loc;
   PVLayerLoc const &postLoc         = *mPostLayer->getLayerLoc();

   int const nx       = preLoc.nx;
   int const ny       = preLoc.ny;
   int const nf       = preLoc.nf;
   int nxPreExtended  = nx + preLoc.halo.lt + preLoc.halo.rt;
   int nyPreExtended  = ny + preLoc.halo.dn + preLoc.halo.up;
   int numPreExtended = nxPreExtended * nyPreExtended * nf;
   pvAssert(numPreExtended * preLoc.nbatch == preActivityCube.numItems);
   int numPostRestricted = nx * ny * nf;

   float *postChannel = mPostLayer->getChannel(mChannelCode);
   for (int b = 0; b < parent->getNBatch(); b++) {
      float const *preActivityBuffer = preActivityCube.data + b * numPreExtended;
      float *postGSynBuffer          = postChannel + b * numPostRestricted;
      if (preActivityCube.isSparse) {
         SparseList<float>::Entry const *activeIndices =
               (SparseList<float>::Entry *)preActivityCube.activeIndices + b * numPreExtended;
         int numActive = preActivityCube.numActive[b];
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
         for (int loopIndex = 0; loopIndex < numActive; loopIndex++) {
            int kPre = activeIndices[loopIndex].index;
            int kx   = kxPos(kPre, nxPreExtended, nyPreExtended, nf) - preLoc.halo.lt;
            int ky   = kyPos(kPre, nxPreExtended, nyPreExtended, nf) - preLoc.halo.up;
            if (kx < 0 or kx >= nx or ky < 0 or kx >= ny) {
               continue;
            }
            int kf    = featureIndex(kPre, nxPreExtended, nyPreExtended, nf);
            int kPost = kIndex(kx, ky, kf, nx, ny, nf);
            pvAssert(kPost >= 0 and kPost < numPostRestricted);
            float a = activeIndices[loopIndex].value;
            postGSynBuffer[kPost] += scale * a;
         }
      }
      else {
         int const nk = postLoc.nx * postLoc.nf;
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
         for (int y = 0; y < ny; y++) {
            int preLineIndex =
                  kIndex(preLoc.halo.lt, y + preLoc.halo.up, 0, nxPreExtended, nyPreExtended, nf);

            float const *preActivityLine = &preActivityBuffer[preLineIndex];
            int postLineIndex            = kIndex(0, y, 0, postLoc.nx, ny, postLoc.nf);
            float *postGSynLine          = &postGSynBuffer[postLineIndex];
            for (int k = 0; k < nk; k++) {
               postGSynLine[k] += scale * preActivityLine[k];
            }
         }
      }
   }
#ifdef PV_USE_CUDA
   mPostLayer->setUpdatedDeviceGSynFlag(!mReceiveGpu);
#endif // PV_USE_CUDA
}

void TopDownDelivery::deliverUnitInput(float *recvBuffer) {

   float scale = 0.0f;
   float prog  = (float)mDeliverCount / mDisplayPeriod;

   if (prog > mZeroRatio) {
      if (prog >= 1.0f - mZeroRatio) {
         scale = 1.0f;
      }
      else {
         // The 0.5f divide by zero is caught above
         scale = (prog - mZeroRatio) / (1.0f - mZeroRatio * 2.0f);
      }
   }

   const int numNeuronsPost = mPostLayer->getNumNeuronsAllBatches();
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
   for (int k = 0; k < numNeuronsPost; k++) {
      recvBuffer[k] += scale;
   }
}

} // end namespace PV

#include <cstring>
