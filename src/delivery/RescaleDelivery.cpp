/*
 * RescaleDelivery.cpp
 *
 *  Created on: Aug 24, 2017
 *      Author: Pete Schultz
 */

#include "RescaleDelivery.hpp"

namespace PV {

RescaleDelivery::RescaleDelivery(char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

void RescaleDelivery::initialize(char const *name, PVParams *params, Communicator const *comm) {
   BaseDelivery::initialize(name, params, comm);
}

void RescaleDelivery::setObjectType() { mObjectType = "RescaleDelivery"; }

int RescaleDelivery::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = IdentDelivery::ioParamsFillGroup(ioFlag);
   ioParam_scale(ioFlag);
   return status;
}

void RescaleDelivery::ioParam_scale(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(
         ioFlag, name, "scale", &mScale, mScale /*default*/, true /*warn if absent*/);
}

// Delivers a scalar multiple of the identity from presynaptic activity to postsynaptic GSyn.
// Rescale::deliver() is largely a duplicate of IdentDelivery::deliver(), except
// for two lines inside for-loops with large numbers of iterations.
// We're discussing ways to eliminate code duplication like this without
// incurring added computational costs.
void RescaleDelivery::deliver(float *destBuffer) {
   if (mChannelCode == CHANNEL_NOUPDATE) {
      return;
   }

   int delay                         = mSingleArbor->getDelay(0);
   PVLayerCube const preActivityCube = mPreData->getPublisher()->createCube(delay);
   PVLayerLoc const &preLoc          = preActivityCube.loc;
   PVLayerLoc const &postLoc         = *mPostGSyn->getLayerLoc();

   int const nx       = preLoc.nx;
   int const ny       = preLoc.ny;
   int const nf       = preLoc.nf;
   int nxPreExtended  = nx + preLoc.halo.lt + preLoc.halo.rt;
   int nyPreExtended  = ny + preLoc.halo.dn + preLoc.halo.up;
   int numPreExtended = nxPreExtended * nyPreExtended * nf;
   pvAssert(numPreExtended * preLoc.nbatch == preActivityCube.numItems);
   int numPostRestricted = nx * ny * nf;

   float *postChannel = destBuffer;
   int const nbatch   = preLoc.nbatch;
   FatalIf(
         postLoc.nbatch != nbatch,
         "%s has different presynaptic and postsynaptic batch sizes.\n",
         getDescription_c());
   for (int b = 0; b < nbatch; b++) {
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
            if (kx < 0 or kx >= nx or ky < 0 or ky >= ny) {
               continue;
            }
            int kf    = featureIndex(kPre, nxPreExtended, nyPreExtended, nf);
            int kPost = kIndex(kx, ky, kf, nx, ny, nf);
            pvAssert(kPost >= 0 and kPost < numPostRestricted);
            float a = activeIndices[loopIndex].value;
            postGSynBuffer[kPost] += mScale * a;
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
               postGSynLine[k] += mScale * preActivityLine[k];
            }
         }
      }
   }
}

void RescaleDelivery::deliverUnitInput(float *recvBuffer) {
   const int numNeuronsPost = mPostGSyn->getBufferSizeAcrossBatch();
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
   for (int k = 0; k < numNeuronsPost; k++) {
      recvBuffer[k] += mScale;
   }
}

} // end namespace PV

#include <cstring>
