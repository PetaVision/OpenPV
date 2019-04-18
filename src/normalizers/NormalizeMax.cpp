/*
 * NormalizeMax.cpp
 *
 *  Created on: Apr 8, 2013
 *      Author: pschultz
 */

#include "NormalizeMax.hpp"

namespace PV {

NormalizeMax::NormalizeMax() { initialize_base(); }

NormalizeMax::NormalizeMax(const char *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

int NormalizeMax::initialize_base() { return PV_SUCCESS; }

int NormalizeMax::initialize(const char *name, HyPerCol *hc) {
   return NormalizeMultiply::initialize(name, hc);
}

int NormalizeMax::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = NormalizeMultiply::ioParamsFillGroup(ioFlag);
   ioParam_minMaxTolerated(ioFlag);
   return status;
}

void NormalizeMax::ioParam_minMaxTolerated(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "minMaxTolerated", &minMaxTolerated, 0.0f, true /*warnIfAbsent*/);
}

int NormalizeMax::normalizeWeights() {
   int status = PV_SUCCESS;

   assert(!mWeightsList.empty());

   // All connections in the group must have the same values of sharedWeights, numArbors, and
   // numDataPatches
   Weights *weights0 = mWeightsList[0];

   float scaleFactor = 1.0f;
   if (mNormalizeFromPostPerspective) {
      if (weights0->getSharedFlag() == false) {
         Fatal().printf(
               "NormalizeMax error for %s: normalizeFromPostPerspective is true but connection "
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

   status = NormalizeMultiply::normalizeWeights(); // applies normalize_cutoff threshold and
   // symmetrizeWeights

   int nArbors        = weights0->getNumArbors();
   int numDataPatches = weights0->getNumDataPatches();
   if (mNormalizeArborsIndividually) {
      for (int arborID = 0; arborID < nArbors; arborID++) {
         for (int patchindex = 0; patchindex < numDataPatches; patchindex++) {
            float max = 0.0f;
            for (auto &weights : mWeightsList) {
               int nxp               = weights->getPatchSizeX();
               int nyp               = weights->getPatchSizeY();
               int nfp               = weights->getPatchSizeF();
               int weightsPerPatch   = nxp * nyp * nfp;
               float *dataStartPatch = weights->getData(arborID) + patchindex * weightsPerPatch;
               accumulateMax(dataStartPatch, weightsPerPatch, &max);
            }
            if (max <= minMaxTolerated) {
               WarnLog().printf(
                     "for NormalizeMax \"%s\": max of weights in patch %d of arbor %d is within "
                     "minMaxTolerated=%f of zero.  Weights in this patch unchanged.\n",
                     getName(),
                     patchindex,
                     arborID,
                     (double)minMaxTolerated);
               continue;
            }
            for (auto &weights : mWeightsList) {
               int nxp               = weights->getPatchSizeX();
               int nyp               = weights->getPatchSizeY();
               int nfp               = weights->getPatchSizeF();
               int weightsPerPatch   = nxp * nyp * nfp;
               float *dataStartPatch = weights->getData(arborID) + patchindex * weightsPerPatch;
               normalizePatch(dataStartPatch, weightsPerPatch, scaleFactor / max);
            }
         }
      }
   }
   else {
      for (int patchindex = 0; patchindex < numDataPatches; patchindex++) {
         float max = 0.0;
         for (int arborID = 0; arborID < nArbors; arborID++) {
            for (auto &weights : mWeightsList) {
               int nxp               = weights->getPatchSizeX();
               int nyp               = weights->getPatchSizeY();
               int nfp               = weights->getPatchSizeF();
               int weightsPerPatch   = nxp * nyp * nfp;
               float *dataStartPatch = weights->getData(arborID) + patchindex * weightsPerPatch;
               accumulateMax(dataStartPatch, weightsPerPatch, &max);
            }
         }
         if (max <= minMaxTolerated) {
            WarnLog().printf(
                  "for NormalizeMax \"%s\": max of weights in patch %d is within "
                  "minMaxTolerated=%f of zero. Weights in this patch unchanged.\n",
                  getName(),
                  patchindex,
                  (double)minMaxTolerated);
            continue;
         }
         for (int arborID = 0; arborID < nArbors; arborID++) {
            for (auto &weights : mWeightsList) {
               int nxp               = weights->getPatchSizeX();
               int nyp               = weights->getPatchSizeY();
               int nfp               = weights->getPatchSizeF();
               int weightsPerPatch   = nxp * nyp * nfp;
               float *dataStartPatch = weights->getData(arborID) + patchindex * weightsPerPatch;
               normalizePatch(dataStartPatch, weightsPerPatch, scaleFactor / max);
            }
         }
      }
   }
   return status;
}

NormalizeMax::~NormalizeMax() {}

} /* namespace PV */
