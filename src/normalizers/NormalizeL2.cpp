/*
 * NormalizeL2.cpp
 *
 *  Created on: Apr 8, 2013
 *      Author: pschultz
 */

#include "NormalizeL2.hpp"

namespace PV {

NormalizeL2::NormalizeL2() { initialize_base(); }

NormalizeL2::NormalizeL2(const char *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

int NormalizeL2::initialize_base() { return PV_SUCCESS; }

int NormalizeL2::initialize(const char *name, HyPerCol *hc) {
   return NormalizeMultiply::initialize(name, hc);
}

int NormalizeL2::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = NormalizeMultiply::ioParamsFillGroup(ioFlag);
   ioParam_minL2NormTolerated(ioFlag);
   return status;
}

void NormalizeL2::ioParam_minL2NormTolerated(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "minL2NormTolerated", &minL2NormTolerated, 0.0f, true /*warnIfAbsent*/);
}

int NormalizeL2::normalizeWeights() {
   int status = PV_SUCCESS;

   assert(!mWeightsList.empty());

   // All connections in the group must have the same values of sharedWeights, numArbors, and
   // numDataPatches
   Weights *weights0 = mWeightsList[0];

   float scaleFactor = 1.0f;
   if (mNormalizeFromPostPerspective) {
      if (weights0->getSharedFlag() == false) {
         Fatal().printf(
               "NormalizeL2 error for %s: normalizeFromPostPerspective is true but connection does "
               "not use shared weights.\n",
               weights0->getName().c_str());
      }
      PVLayerLoc const &preLoc  = weights0->getGeometry()->getPreLoc();
      PVLayerLoc const &postLoc = weights0->getGeometry()->getPostLoc();
      int numNeuronsPre         = preLoc.nx * preLoc.ny * preLoc.nf;
      int numNeuronsPost        = postLoc.nx * postLoc.ny * postLoc.nf;
      scaleFactor               = ((float)numNeuronsPost) / ((float)numNeuronsPre);
   }
   scaleFactor *= mStrength;

   status = NormalizeMultiply::normalizeWeights(); // applies normalize_cutoff threshold and
   // rMinX,rMinY

   int nArbors        = weights0->getNumArbors();
   int numDataPatches = weights0->getNumDataPatches();
   if (mNormalizeArborsIndividually) {
      for (int arborID = 0; arborID < nArbors; arborID++) {
         for (int patchindex = 0; patchindex < numDataPatches; patchindex++) {
            float sumsq = 0.0f;
            for (auto &weights : mWeightsList) {
               int nxp               = weights->getPatchSizeX();
               int nyp               = weights->getPatchSizeY();
               int nfp               = weights->getPatchSizeF();
               int weightsPerPatch   = nxp * nyp * nfp;
               float *dataStartPatch = weights->getData(arborID) + patchindex * weightsPerPatch;
               accumulateSumSquared(dataStartPatch, weightsPerPatch, &sumsq);
            }
            float l2norm = sqrtf(sumsq);
            if (fabsf(l2norm) <= minL2NormTolerated) {
               WarnLog().printf(
                     "for NormalizeL2 \"%s\": sum of squares of weights in patch %d of arbor %d is "
                     "within minL2NormTolerated=%f of zero.  Weights in this patch unchanged.\n",
                     getName(),
                     patchindex,
                     arborID,
                     (double)minL2NormTolerated);
               continue;
            }
            for (auto &weights : mWeightsList) {
               int nxp               = weights->getPatchSizeX();
               int nyp               = weights->getPatchSizeY();
               int nfp               = weights->getPatchSizeF();
               int weightsPerPatch   = nxp * nyp * nfp;
               float *dataStartPatch = weights->getData(arborID) + patchindex * weightsPerPatch;
               normalizePatch(dataStartPatch, weightsPerPatch, scaleFactor / l2norm);
            }
         }
      }
   }
   else {
      for (int patchindex = 0; patchindex < numDataPatches; patchindex++) {
         float sumsq = 0.0f;
         for (int arborID = 0; arborID < nArbors; arborID++) {
            for (auto &weights : mWeightsList) {
               int nxp               = weights->getPatchSizeX();
               int nyp               = weights->getPatchSizeY();
               int nfp               = weights->getPatchSizeF();
               int xPatchStride      = weights->getPatchStrideX();
               int yPatchStride      = weights->getPatchStrideY();
               int weightsPerPatch   = nxp * nyp * nfp;
               float *dataStartPatch = weights->getData(arborID) + patchindex * weightsPerPatch;
               accumulateSumSquared(dataStartPatch, weightsPerPatch, &sumsq);
            }
         }
         float l2norm = sqrtf(sumsq);
         if (fabsf(sumsq) <= minL2NormTolerated) {
            WarnLog().printf(
                  "for NormalizeL2 \"%s\": sum of squares of weights in patch %d is within "
                  "minL2NormTolerated=%f of zero.  Weights in this patch unchanged.\n",
                  getName(),
                  patchindex,
                  (double)minL2NormTolerated);
            break;
         }
         for (int arborID = 0; arborID < nArbors; arborID++) {
            for (auto &weights : mWeightsList) {
               int nxp               = weights->getPatchSizeX();
               int nyp               = weights->getPatchSizeY();
               int nfp               = weights->getPatchSizeF();
               int weightsPerPatch   = nxp * nyp * nfp;
               float *dataStartPatch = weights->getData(arborID) + patchindex * weightsPerPatch;
               normalizePatch(dataStartPatch, weightsPerPatch, scaleFactor / l2norm);
            }
         }
      }
   }
   return status;
}

NormalizeL2::~NormalizeL2() {}

} /* namespace PV */
