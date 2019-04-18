/*
 * NormalizeSum.cpp
 *
 *  Created on: Apr 8, 2013
 *      Author: pschultz
 */

#include "NormalizeSum.hpp"
#include <iostream>

namespace PV {

NormalizeSum::NormalizeSum() { initialize_base(); }

NormalizeSum::NormalizeSum(const char *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

NormalizeSum::~NormalizeSum() {}

int NormalizeSum::initialize_base() { return PV_SUCCESS; }

int NormalizeSum::initialize(const char *name, HyPerCol *hc) {
   return NormalizeMultiply::initialize(name, hc);
}

int NormalizeSum::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = NormalizeMultiply::ioParamsFillGroup(ioFlag);
   ioParam_minSumTolerated(ioFlag);
   return status;
}

void NormalizeSum::ioParam_minSumTolerated(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag,
         name,
         "minSumTolerated",
         &mMinSumTolerated,
         mMinSumTolerated,
         true /*warnIfAbsent*/);
}

int NormalizeSum::normalizeWeights() {
   int status = PV_SUCCESS;

   pvAssert(!mWeightsList.empty());

   // All connections in the group must have the same values of sharedWeights, numArbors, and
   // numDataPatches
   Weights *weights0 = mWeightsList[0];

   float scaleFactor = 1.0f;
   if (mNormalizeFromPostPerspective) {
      if (weights0->getSharedFlag() == false) {
         Fatal().printf(
               "NormalizeSum error for %s: normalizeFromPostPerspective is true but connection "
               "does not use shared weights.\n",
               getDescription_c());
      }
      PVLayerLoc const &preLoc  = weights0->getGeometry()->getPreLoc();
      PVLayerLoc const &postLoc = weights0->getGeometry()->getPostLoc();
      int numNeuronsPre         = preLoc.nx * preLoc.ny * preLoc.nf;
      int numNeuronsPost        = postLoc.nx * postLoc.ny * postLoc.nf;
      scaleFactor               = ((float)numNeuronsPost) / ((float)numNeuronsPre);
   }
   scaleFactor *= mStrength;

   status = NormalizeBase::normalizeWeights(); // applies normalize_cutoff threshold and
   // symmetrizeWeights

   int nArbors        = weights0->getNumArbors();
   int numDataPatches = weights0->getNumDataPatches();
   if (mNormalizeArborsIndividually) {
      for (int arborID = 0; arborID < nArbors; arborID++) {
         for (int patchindex = 0; patchindex < numDataPatches; patchindex++) {
            float sum = 0.0;
            for (auto &weights : mWeightsList) {
               int nxp               = weights->getPatchSizeX();
               int nyp               = weights->getPatchSizeY();
               int nfp               = weights->getPatchSizeF();
               int weightsPerPatch   = nxp * nyp * nfp;
               float *dataStartPatch = weights->getData(arborID) + patchindex * weightsPerPatch;
               accumulateSum(dataStartPatch, weightsPerPatch, &sum);
            }
            if (fabsf(sum) <= mMinSumTolerated) {
               WarnLog().printf(
                     "NormalizeSum for %s: sum of weights in patch %d of arbor %d is within "
                     "minSumTolerated=%f of zero. Weights in this patch unchanged.\n",
                     getDescription_c(),
                     patchindex,
                     arborID,
                     (double)mMinSumTolerated);
               continue;
            }
            for (auto &weights : mWeightsList) {
               int nxp               = weights->getPatchSizeX();
               int nyp               = weights->getPatchSizeY();
               int nfp               = weights->getPatchSizeF();
               int weightsPerPatch   = nxp * nyp * nfp;
               float *dataStartPatch = weights->getData(arborID) + patchindex * weightsPerPatch;
               normalizePatch(dataStartPatch, weightsPerPatch, scaleFactor / sum);
            }
         }
      }
   }
   else {
      for (int patchindex = 0; patchindex < numDataPatches; patchindex++) {
         float sum = 0.0;
         for (int arborID = 0; arborID < nArbors; arborID++) {
            for (auto &weights : mWeightsList) {
               int nxp               = weights->getPatchSizeX();
               int nyp               = weights->getPatchSizeY();
               int nfp               = weights->getPatchSizeF();
               int weightsPerPatch   = nxp * nyp * nfp;
               float *dataStartPatch = weights->getData(arborID) + patchindex * weightsPerPatch;
               accumulateSum(dataStartPatch, weightsPerPatch, &sum);
            }
         }
         if (fabsf(sum) <= mMinSumTolerated) {
            WarnLog().printf(
                  "NormalizeSum for %s: sum of weights in patch %d is within minSumTolerated=%f of "
                  "zero.  Weights in this patch unchanged.\n",
                  getDescription_c(),
                  patchindex,
                  (double)mMinSumTolerated);
            continue;
         }
         for (int arborID = 0; arborID < nArbors; arborID++) {
            for (auto &weights : mWeightsList) {
               int nxp               = weights->getPatchSizeX();
               int nyp               = weights->getPatchSizeY();
               int nfp               = weights->getPatchSizeF();
               int weightsPerPatch   = nxp * nyp * nfp;
               float *dataStartPatch = weights->getData(arborID) + patchindex * weightsPerPatch;
               normalizePatch(dataStartPatch, weightsPerPatch, scaleFactor / sum);
            }
         }
      } // patchindex
   } // mNormalizeArborsIndividually
   return status;
}

} /* namespace PV */
