/*
 * PlasticTestUpdater.cpp
 *
 *  Created on: Oct 19, 2011
 *      Author: pschultz
 */

#include "PlasticTestUpdater.hpp"

namespace PV {

PlasticTestUpdater::PlasticTestUpdater(const char *name, PVParams *params, Communicator const *comm)
      : HebbianUpdater() {
   HebbianUpdater::initialize(name, params, comm);
}

int PlasticTestUpdater::update_dW(int arborID) {
   // compute dW but don't add them to the weights yet.
   // That takes place in reduceKernels, so that the output is
   // independent of the number of processors.
   HyPerLayer *pre           = mConnectionData->getPre();
   HyPerLayer *post          = mConnectionData->getPost();
   long nExt                 = pre->getNumExtended();
   PVLayerLoc const *preLoc  = pre->getLayerLoc();
   PVLayerLoc const *postLoc = post->getLayerLoc();
   int const nbatch          = preLoc->nbatch;
   int delay                 = mArborList->getDelay(arborID);

   float const *preactbufHead =
         pre->getComponentByType<BasePublisherComponent>()->getLayerData(delay);
   float const *postactbufHead = post->getComponentByType<BasePublisherComponent>()->getLayerData();

   if (mWeights->getSharedWeightsFlag()) {
      // Calculate x and y cell size
      int xCellSize   = zUnitCellSize(preLoc->nx, postLoc->nx);
      int yCellSize   = zUnitCellSize(preLoc->ny, postLoc->ny);
      int nxExt       = preLoc->nx + preLoc->halo.lt + preLoc->halo.rt;
      int nyExt       = preLoc->ny + preLoc->halo.up + preLoc->halo.dn;
      int nf          = preLoc->nf;
      long numKernels = mWeights->getNumDataPatchesOverall();

      for (int b = 0; b < nbatch; b++) {
// Shared weights done in parallel, parallel in numkernels
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
         for (long kernelIdx = 0; kernelIdx < numKernels; kernelIdx++) {

            // Calculate xCellIdx, yCellIdx, and fCellIdx from kernelIndex
            int kxCellIdx = kxPos(kernelIdx, xCellSize, yCellSize, nf);
            int kyCellIdx = kyPos(kernelIdx, xCellSize, yCellSize, nf);
            int kfIdx     = featureIndex(kernelIdx, xCellSize, yCellSize, nf);
            // Loop over all cells in pre ext
            int kyIdx    = kyCellIdx;
            int yCellIdx = 0;
            while (kyIdx < nyExt) {
               int kxIdx    = kxCellIdx;
               int xCellIdx = 0;
               while (kxIdx < nxExt) {
                  // Calculate kExt from ky, kx, and kf
                  long kExt = kIndex(kxIdx, kyIdx, kfIdx, nxExt, nyExt, nf);
                  modifiedUpdateInd_dW(arborID, b, preactbufHead, postactbufHead, kExt);
                  xCellIdx++;
                  kxIdx = kxCellIdx + xCellIdx * xCellSize;
               }
               yCellIdx++;
               kyIdx = kyCellIdx + yCellIdx * yCellSize;
            }
         }
      }
   }
   else {
      if (mNormalizeDw) {
         for (long int &a : mNumPatchActivations[arborID]) { a = 0.0f; }
      }
// Shared weights done in parallel, parallel in numkernels
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for collapse(2)
#endif
      for (int b = 0; b < nbatch; b++) {
         for (long kExt = 0; kExt < nExt; kExt++) {
            modifiedUpdateInd_dW(arborID, b, preactbufHead, postactbufHead, kExt);
         }
      }
   }

   // If update from clones, update dw here as well
   // Updates on all PlasticClones
   for (auto &c : mClones) {
      HyPerLayer *clonePreLayer  = c->getPre();
      HyPerLayer *clonePostLayer = c->getPost();
      auto *clonePrePublisher    = clonePreLayer->getComponentByType<BasePublisherComponent>();
      auto *clonePostPublisher   = clonePostLayer->getComponentByType<BasePublisherComponent>();
      pvAssert(clonePrePublisher->getNumExtended() == nExt);
      pvAssert(clonePrePublisher->getLayerLoc()->nbatch == nbatch);
      float const *clonePre  = clonePrePublisher->getLayerData(delay);
      float const *clonePost = clonePostPublisher->getLayerData();
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for collapse(2)
#endif
      for (int b = 0; b < nbatch; b++) {
         for (long kExt = 0; kExt < nExt; kExt++) {
            modifiedUpdateInd_dW(arborID, b, clonePre, clonePost, kExt);
         }
      }
   }

   return PV_SUCCESS;
}

void PlasticTestUpdater::modifiedUpdateInd_dW(
      int arborID,
      int batchID,
      float const *preLayerData,
      float const *postLayerData,
      long kExt) {
   HyPerLayer *pre           = mConnectionData->getPre();
   HyPerLayer *post          = mConnectionData->getPost();
   const PVLayerLoc *postLoc = post->getLayerLoc();

   const float *preactbuf  = preLayerData + batchID * pre->getNumExtended();
   const float *postactbuf = postLayerData + batchID * post->getNumExtended();

   int sya = (postLoc->nf * (postLoc->nx + postLoc->halo.lt + postLoc->halo.rt));

   float preact = preactbuf[kExt];
   if (preact == 0.0f) {
      return;
   }

   Patch const &patch = mWeights->getPatch(kExt);
   int ny             = patch.ny;
   int nk             = patch.nx * mWeights->getPatchSizeF();
   if (ny == 0 || nk == 0) {
      return;
   }

   size_t offset           = mWeights->getGeometry()->getAPostOffset(kExt);
   const float *postactRef = &postactbuf[offset];

   float *dwdata =
         mDeltaWeights->getDataFromPatchIndex(arborID, kExt) + mDeltaWeights->getPatch(kExt).offset;
   long *activations = nullptr;
   if (mNormalizeDw) {
      if (mWeights->getSharedWeightsFlag()) {
         long dataIndex        = mWeights->calcDataIndexFromPatchIndex(kExt);
         long patchSizeOverall = mWeights->getPatchSizeOverall();
         int patchOffset       = (int)mWeights->getPatch(kExt).offset;
         activations = &mNumKernelActivations[arborID][dataIndex * patchSizeOverall + patchOffset];
      }
      else {
         mNumPatchActivations[arborID][kExt]++;
      }
   }

   int syp         = mWeights->getPatchStrideY();
   int lineoffsetw = 0;
   int lineoffseta = 0;
   // Can't parallelize this because this function is called in a loop over kExt,
   // and that loop is parallelized
   for (int y = 0; y < ny; y++) {
      for (int k = 0; k < nk; k++) {
         float aPost = postactRef[lineoffseta + k];
         // calculate contribution to dw
         // Note: this is a hack, as batching calls this function, but overwrites to allocate
         // numKernelActivations with non-shared weights
         if (activations and mWeights->getSharedWeightsFlag()) {
            // Offset in the case of a shrunken patch, where dwdata is applying when calling
            // getDeltaWeightsData
            activations[lineoffsetw + k]++;
         }
         // Modified update rule
         dwdata[lineoffsetw + k] += preact - aPost;
      }
      lineoffsetw += syp;
      lineoffseta += sya;
   }
   return;
}

PlasticTestUpdater::~PlasticTestUpdater() {}

} /* namespace PV */
