/*
 * TransposeWeights.cpp
 *
 *  Created on: Sep 1, 2017
 *      Author: peteschultz
 */

#include "TransposeWeights.hpp"
#include "utils/BorderExchange.hpp"
#include "utils/PVAssert.hpp"
#include "utils/PVLog.hpp"
#include "utils/conversions.h"

namespace PV {

void TransposeWeights::transpose(Weights *preWeights, Weights *postWeights, Communicator *comm) {
   int const numArbors = preWeights->getNumArbors();
   FatalIf(
         numArbors != postWeights->getNumArbors(),
         "transpose called from weights \"%s\" to weights \"%s\", "
         "but these do not have the same number of arbors (%d versus %d).\n",
         preWeights->getName().c_str(),
         postWeights->getName().c_str(),
         preWeights->getNumArbors(),
         postWeights->getNumArbors());
   for (int arborIndex = 0; arborIndex < numArbors; arborIndex++) {
      transpose(preWeights, postWeights, comm, arborIndex);
   }
}

void TransposeWeights::transpose(
      Weights *preWeights,
      Weights *postWeights,
      Communicator *comm,
      int arbor) {
   // TODO: Check if preWeights's preLoc is postWeights's postLoc and vice versa
   bool sharedFlag = preWeights->getSharedFlag();
   FatalIf(
         postWeights->getSharedFlag() != sharedFlag,
         "Transposing weights %s to %s, but SharedFlag values do not match.\n",
         preWeights->getName().c_str(),
         postWeights->getName().c_str());
   // Note: if preWeights->sharedFlag is true and postWeights->sharedFlag is false,
   // the transpose operation is well-defined; we just haven't had occasion to use that case.
   if (sharedFlag) {
      transposeShared(preWeights, postWeights, arbor);
   }
   else {
      transposeNonshared(preWeights, postWeights, comm, arbor);
   }
}

void TransposeWeights::transposeShared(Weights *preWeights, Weights *postWeights, int arbor) {
   int const numPatchesXPre = preWeights->getNumDataPatchesX();
   int const numPatchesYPre = preWeights->getNumDataPatchesY();
   int const numPatchesFPre = preWeights->getNumDataPatchesF();
   int const numPatchesPre  = preWeights->getNumDataPatches();
   int const patchSizeXPre  = preWeights->getPatchSizeX();
   int const patchSizeYPre  = preWeights->getPatchSizeY();
   int const patchSizeFPre  = preWeights->getPatchSizeF();
   int const patchSizePre   = patchSizeXPre * patchSizeYPre * patchSizeFPre;

   int const numPatchesXPost = postWeights->getNumDataPatchesX();
   int const numPatchesYPost = postWeights->getNumDataPatchesY();
   int const numPatchesFPost = postWeights->getNumDataPatchesF();

#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for collapse(2)
#endif
   for (int patchIndexPre = 0; patchIndexPre < numPatchesPre; patchIndexPre++) {
      for (int itemInPatchPre = 0; itemInPatchPre < patchSizePre; itemInPatchPre++) {
         int const patchIndexXPre =
               kxPos(patchIndexPre, numPatchesXPre, numPatchesYPre, numPatchesFPre);
         int const patchIndexYPre =
               kyPos(patchIndexPre, numPatchesXPre, numPatchesYPre, numPatchesFPre);
         int const patchIndexFPre =
               featureIndex(patchIndexPre, numPatchesXPre, numPatchesYPre, numPatchesFPre);

         int itemInPatchPost =
               preWeights->getGeometry()->getTransposeItemIndex(patchIndexPre, itemInPatchPre);

         int patchIndexXPost = 0;
         if (numPatchesXPost > 1) { // one-to-many from presynaptic perspective
            int const itemInPatchXPre =
                  kxPos(itemInPatchPre, patchSizeXPre, patchSizeYPre, patchSizeFPre);
            patchIndexXPost = -(patchSizeXPre - numPatchesXPost) / 2 + itemInPatchXPre;
            patchIndexXPost %= numPatchesXPost;
            if (patchIndexXPost < 0) {
               patchIndexXPost += numPatchesXPost;
            }
         }

         int patchIndexYPost = 0;
         if (numPatchesYPost > 1) { // one-to-many from presynaptic perspective
            int const itemInPatchYPre =
                  kyPos(itemInPatchPre, patchSizeXPre, patchSizeYPre, patchSizeFPre);
            patchIndexYPost = -(patchSizeYPre - numPatchesYPost) / 2 + itemInPatchYPre;
            patchIndexYPost %= numPatchesYPost;
            if (patchIndexYPost < 0) {
               patchIndexYPost += numPatchesYPost;
            }
         }

         int const itemInPatchFPre =
               featureIndex(itemInPatchPre, patchSizeXPre, patchSizeYPre, patchSizeFPre);
         int patchIndexFPost = itemInPatchFPre;

         int patchIndexPost = kIndex(
               patchIndexXPost,
               patchIndexYPost,
               patchIndexFPost,
               numPatchesXPost,
               numPatchesYPost,
               numPatchesFPost);
         postWeights->getDataFromDataIndex(arbor, patchIndexPost)[itemInPatchPost] =
               preWeights->getDataFromDataIndex(arbor, patchIndexPre)[itemInPatchPre];
      }
   }
}

void TransposeWeights::transposeNonshared(
      Weights *preWeights,
      Weights *postWeights,
      Communicator *comm,
      int arbor) {
   int const numPatchesXPre = preWeights->getNumDataPatchesX();
   int const numPatchesYPre = preWeights->getNumDataPatchesY();
   int const numPatchesFPre = preWeights->getNumDataPatchesF();
   int const numPatchesPre  = preWeights->getNumDataPatches();

   int const patchSizeXPre = preWeights->getPatchSizeX();
   int const patchSizeYPre = preWeights->getPatchSizeY();
   int const patchSizeFPre = preWeights->getPatchSizeF();
   int const patchSizePre  = patchSizeXPre * patchSizeYPre * patchSizeFPre;

   int const numPatchesXPost = postWeights->getNumDataPatchesX();
   int const numPatchesYPost = postWeights->getNumDataPatchesY();
   int const numPatchesFPost = postWeights->getNumDataPatchesF();
   int const numPatchesPost  = postWeights->getNumDataPatches();
   int const patchSizePost   = postWeights->getPatchSizeOverall();

   PVLayerLoc const &preLoc  = preWeights->getGeometry()->getPreLoc();
   PVLayerLoc const &postLoc = postWeights->getGeometry()->getPreLoc();

   int const nxPre            = preLoc.nx;
   int const nyPre            = preLoc.ny;
   int const nfPre            = preLoc.nf;
   int const numRestrictedPre = nxPre * nyPre * nfPre;

   int const nxPost            = postLoc.nx;
   int const nyPost            = postLoc.ny;
   int const nfPost            = postLoc.nf;
   int const numRestrictedPost = nxPost * nyPost * nfPost;

   int const numKernelsXPre = preWeights->getGeometry()->getNumKernelsX();
   int const numKernelsYPre = preWeights->getGeometry()->getNumKernelsY();
   int const numKernelsFPre = preWeights->getGeometry()->getNumKernelsF();

   std::size_t const numPostWeightValues = (std::size_t)(numPatchesPost * patchSizePost);
   memset(postWeights->getDataFromDataIndex(arbor, 0), 0, numPostWeightValues * sizeof(float));

#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for collapse(2)
#endif
   for (int patchIndexPre = 0; patchIndexPre < numPatchesPre; patchIndexPre++) {
      for (int itemInPatchPre = 0; itemInPatchPre < patchSizePre; itemInPatchPre++) {
         int const patchIndexXPre =
               kxPos(patchIndexPre, numPatchesXPre, numPatchesYPre, numPatchesFPre);
         int kernelIndexXPre = (patchIndexXPre - preLoc.halo.lt) % numKernelsXPre;
         if (kernelIndexXPre < 0) {
            kernelIndexXPre += numKernelsXPre;
         }
         pvAssert(kernelIndexXPre >= 0 and kernelIndexXPre < numKernelsXPre);

         int const patchIndexYPre =
               kyPos(patchIndexPre, numPatchesXPre, numPatchesYPre, numPatchesFPre);
         int kernelIndexYPre = (patchIndexYPre - preLoc.halo.up) % numKernelsYPre;
         if (kernelIndexYPre < 0) {
            kernelIndexYPre += numKernelsYPre;
         }
         pvAssert(kernelIndexYPre >= 0 and kernelIndexYPre < numKernelsYPre);

         int const patchIndexFPre =
               featureIndex(patchIndexPre, numPatchesXPre, numPatchesYPre, numPatchesFPre);
         int const kernelIndexFPre = patchIndexFPre;

         int const kernelIndexPre = kIndex(
               kernelIndexXPre,
               kernelIndexYPre,
               kernelIndexFPre,
               numKernelsXPre,
               numKernelsYPre,
               numKernelsFPre);
         int const itemInPatchPost =
               preWeights->getGeometry()->getTransposeItemIndex(kernelIndexPre, itemInPatchPre);

         Patch const &patch     = preWeights->getPatch(patchIndexPre);
         int const patchOffsetX = kxPos(patch.offset, patchSizeXPre, patchSizeYPre, patchSizeFPre);
         int const itemInPatchXPre =
               kxPos(itemInPatchPre, patchSizeXPre, patchSizeYPre, patchSizeFPre);
         if (itemInPatchXPre < patchOffsetX or itemInPatchXPre >= patchOffsetX + patch.nx) {
            continue;
         }

         int const patchOffsetY = kyPos(patch.offset, patchSizeXPre, patchSizeYPre, patchSizeFPre);
         int const itemInPatchYPre =
               kyPos(itemInPatchPre, patchSizeXPre, patchSizeYPre, patchSizeFPre);
         if (itemInPatchYPre < patchOffsetY or itemInPatchYPre >= patchOffsetY + patch.ny) {
            continue;
         }

         int const aPostOffset  = preWeights->getGeometry()->getAPostOffset(patchIndexPre);
         int const aPostOffsetX = kxPos(aPostOffset, numPatchesXPost, numPatchesYPost, nfPost);
         int const aPostOffsetY = kyPos(aPostOffset, numPatchesXPost, numPatchesYPost, nfPost);

         int const patchIndexXPost = aPostOffsetX + itemInPatchXPre - patchOffsetX;
         int const patchIndexYPost = aPostOffsetY + itemInPatchYPre - patchOffsetY;
         int const patchIndexFPost =
               featureIndex(itemInPatchPre, patchSizeXPre, patchSizeYPre, patchSizeFPre);

         int patchIndexPost = kIndex(
               patchIndexXPost,
               patchIndexYPost,
               patchIndexFPost,
               numPatchesXPost,
               numPatchesYPost,
               numPatchesFPost);
         pvAssert(patchIndexPost >= 0 and patchIndexPost < postWeights->getNumDataPatches());
         postWeights->getDataFromDataIndex(arbor, patchIndexPost)[itemInPatchPost] =
               preWeights->getDataFromDataIndex(arbor, patchIndexPre)[itemInPatchPre];
      }
   }

   PVLayerLoc transposeLoc;
   memcpy(&transposeLoc, &postLoc, sizeof(transposeLoc));
   transposeLoc.nf = postLoc.nf * patchSizePost;

   BorderExchange borderExchange(*comm->getLocalMPIBlock(), transposeLoc);
   float *data = postWeights->getDataFromDataIndex(arbor, 0);
   std::vector<MPI_Request> mpiRequest;
   pvAssert(mpiRequest.size() == (std::size_t)0);
   borderExchange.exchange(data, mpiRequest);

   // blocks on MPI communication; should separate out the wait to provide concurrency.
   borderExchange.wait(mpiRequest);

   int const patchSizeXPost = postWeights->getPatchSizeX();
   int const patchSizeYPost = postWeights->getPatchSizeY();
   int const patchSizeFPost = postWeights->getPatchSizeF();
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for collapse(2)
#endif
   for (int patchIndexPost = 0; patchIndexPost < numPatchesPost; patchIndexPost++) {
      for (int itemInPatchPost = 0; itemInPatchPost < patchSizePost; itemInPatchPost++) {
         Patch const &patchPost = postWeights->getPatch(patchIndexPost);

         int const patchOffsetXPost =
               kxPos(patchPost.offset, patchSizeXPost, patchSizeYPost, patchSizeFPost);
         int const itemInPatchXPost =
               kxPos(itemInPatchPost, patchSizeXPost, patchSizeYPost, patchSizeFPost);
         int const fromOffsetXPost   = itemInPatchXPost - patchOffsetXPost;
         bool const xInShrunkenPatch = fromOffsetXPost >= 0 and fromOffsetXPost < patchPost.nx;

         int const patchOffsetYPost =
               kyPos(patchPost.offset, patchSizeXPost, patchSizeYPost, patchSizeFPost);
         int const itemInPatchYPost =
               kyPos(itemInPatchPost, patchSizeXPost, patchSizeYPost, patchSizeFPost);
         int const fromOffsetYPost   = itemInPatchYPost - patchOffsetYPost;
         bool const yInShrunkenPatch = fromOffsetYPost >= 0 and fromOffsetYPost < patchPost.ny;

         if (!xInShrunkenPatch or !yInShrunkenPatch) {
            postWeights->getDataFromDataIndex(arbor, patchIndexPost)[itemInPatchPost] = 0.0f;
         }
      }
   }
}

} // namespace PV
