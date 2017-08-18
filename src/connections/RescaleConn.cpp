/*
 * RescaleConn.cpp
 *
 *  Created on: Apr 15, 2016
 *      Author: pschultz
 */

#include "RescaleConn.hpp"

namespace PV {

RescaleConn::RescaleConn(char const *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

RescaleConn::RescaleConn() { initialize_base(); }

int RescaleConn::initialize_base() {
   scale = 1.0f;
   return PV_SUCCESS;
}

int RescaleConn::initialize(char const *name, HyPerCol *hc) {
   return IdentConn::initialize(name, hc);
}

int RescaleConn::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = IdentConn::ioParamsFillGroup(ioFlag);
   ioParam_scale(ioFlag);
   return status;
}

void RescaleConn::ioParam_scale(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "scale", &scale, scale /*default*/, true /*warn if absent*/);
}

int RescaleConn::deliver() {
   // Largely a duplicate of IdentConn::deliver, except
   // for two lines inside for-loops with large numbers of iterations.
   // We're discussing ways to eliminate code duplication like this without
   // incurring added computational costs.  For now, leaving the duplicate
   // code as is.  --peteschultz, April 15, 2016 (edited Aug 17, 2017).

   if (getChannel() == CHANNEL_NOUPDATE) {
      return PV_SUCCESS;
   }
   float *postChannel = post->getChannel(getChannel());
   pvAssert(postChannel);

   pvAssert(numberOfAxonalArborLists() == 1);

   int const delay                   = getDelay(0);
   HyPerLayer *pre                   = preSynapticLayer();
   PVLayerCube const preActivityCube = pre->getPublisher()->createCube(delay);

   HyPerLayer *post = postSynapticLayer();
   pvAssert(pre->getNumNeurons() == post->getNumNeurons());

   PVLayerLoc const *preLoc  = &preActivityCube.loc;
   PVLayerLoc const *postLoc = post->getLayerLoc();
   pvAssert(preLoc->nx == postLoc->nx and preLoc->ny == postLoc->ny and preLoc->nf == postLoc->nf);
   int const nx       = preLoc->nx;
   int const ny       = preLoc->ny;
   int const nf       = preLoc->nf;
   int nxPreExtended  = nx + preLoc->halo.lt + preLoc->halo.rt;
   int nyPreExtended  = ny + preLoc->halo.dn + preLoc->halo.up;
   int numPreExtended = nxPreExtended * nyPreExtended * nf;
   pvAssert(numPreExtended == pre->getNumExtended());

   for (int b = 0; b < parent->getNBatch(); b++) {
      float const *preActivityBuffer = preActivityCube.data + b * numPreExtended;
      float *postGSynBuffer          = postChannel + b * post->getNumNeurons();
      if (preActivityCube.isSparse) {
         SparseList<float>::Entry const *activeIndices =
               (SparseList<float>::Entry *)preActivityCube.activeIndices + b * numPreExtended;
         int numActive = preActivityCube.numActive[b];
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
         for (int loopIndex = 0; loopIndex < numActive; loopIndex++) {
            int kPre = activeIndices[loopIndex].index;
            int kx   = kxPos(kPre, nxPreExtended, nyPreExtended, nf) - preLoc->halo.lt;
            int ky   = kyPos(kPre, nxPreExtended, nyPreExtended, nf) - preLoc->halo.up;
            if (kx < 0 or kx >= nx or ky < 0 or kx >= ny) {
               continue;
            }
            int kf    = featureIndex(kPre, nxPreExtended, nyPreExtended, nf) - preLoc->halo.up;
            int kPost = kIndex(kx, ky, kf, nx, ny, nf);
            pvAssert(kPost >= 0 and kPost < post->getNumNeurons());
            float a = scale * activeIndices[loopIndex].value;
            postGSynBuffer[kPost] += a;
         }
      }
      else {
         int const nk = postLoc->nx * postLoc->nf;
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
         for (int y = 0; y < ny; y++) {
            int preLineIndex =
                  kIndex(preLoc->halo.lt, y + preLoc->halo.up, 0, nxPreExtended, nyPreExtended, nf);

            float const *preActivityLine = &preActivityBuffer[preLineIndex];
            int postLineIndex            = kIndex(0, y, 0, postLoc->nx, ny, postLoc->nf);
            float *postGSynLine          = &postGSynBuffer[postLineIndex];
            for (int k = 0; k < nk; k++) {
               postGSynLine[k] += scale * preActivityLine[k];
            }
         }
      }
   }
   return PV_SUCCESS;
}

RescaleConn::~RescaleConn() {}

} /* namespace PV */
