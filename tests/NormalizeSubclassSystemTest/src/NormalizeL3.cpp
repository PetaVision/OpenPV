/*
 * NormalizeL3.cpp
 */

#include "NormalizeL3.hpp"
#include <columns/HyPerCol.hpp>

namespace PV {

NormalizeL3::NormalizeL3(char const *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

NormalizeL3::NormalizeL3() { initialize_base(); }

int NormalizeL3::initialize_base() {
   minL3NormTolerated = 0.0f;
   return PV_SUCCESS;
}

int NormalizeL3::initialize(char const *name, HyPerCol *hc) {
   return NormalizeMultiply::initialize(name, hc);
}

int NormalizeL3::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = NormalizeMultiply::ioParamsFillGroup(ioFlag);
   ioParam_minL3NormTolerated(ioFlag);
   return status;
}

void NormalizeL3::ioParam_minL3NormTolerated(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag,
         name,
         "minL3NormTolerated",
         &minL3NormTolerated,
         minL3NormTolerated,
         true /*warnIfAbsent*/);
}
int NormalizeL3::normalizeWeights() {
   int status = PV_SUCCESS;

   FatalIf(mWeightsList.empty(), "normalizeWeights called with weightsList empty.\n");

   // All connections in the group must have the same values of sharedWeights, numArbors, and
   // numDataPatches
   Weights *weights0 = mWeightsList[0];

   float scaleFactor = 1.0f;
   if (mNormalizeFromPostPerspective) {
      if (weights0->getSharedFlag() == false) {
         Fatal().printf(
               "NormalizeL3 error for %s: normalizeFromPostPerspective is true but connection does "
               "not use shared weights.\n",
               weights0->getName().c_str());
      }
      PVLayerLoc const &preLoc  = weights0->getGeometry()->getPreLoc();
      PVLayerLoc const &postLoc = weights0->getGeometry()->getPostLoc();
      int numNeuronsPre         = preLoc.nx * preLoc.ny * preLoc.nf;
      int numNeuronsPost        = postLoc.nx * postLoc.ny * postLoc.nf;
      scaleFactor               = ((float)numNeuronsPre) / ((float)numNeuronsPost);
   }
   scaleFactor *= mStrength;

   status = NormalizeMultiply::normalizeWeights(); // applies normalize_cutoff threshold and
   // rMinX,rMinY

   int nArbors        = weights0->getNumArbors();
   int numDataPatches = weights0->getNumDataPatches();
   if (mNormalizeArborsIndividually) {
      for (int arborID = 0; arborID < nArbors; arborID++) {
         for (int patchindex = 0; patchindex < numDataPatches; patchindex++) {
            float sumcubed = 0.0f;
            for (auto &weights : mWeightsList) {
               int nxp               = weights->getPatchSizeX();
               int nyp               = weights->getPatchSizeY();
               int nfp               = weights->getPatchSizeF();
               int xPatchStride      = weights->getPatchStrideX();
               int yPatchStride      = weights->getPatchStrideY();
               int weights_per_patch = nxp * nyp * nfp;
               float *dataStartPatch = weights->getData(arborID) + patchindex * weights_per_patch;
               for (int k = 0; k < weights_per_patch; k++) {
                  float w = fabs(dataStartPatch[k]);
                  sumcubed += w * w * w;
               }
            }
            float l3norm = powf(sumcubed, 1.0f / 3.0f);
            if (fabsf(l3norm) <= minL3NormTolerated) {
               WarnLog().printf(
                     "NormalizeL3 \"%s\": L^3 norm in patch %d of arbor %d is within "
                     "minL3NormTolerated=%f of zero.  Weights in this patch unchanged.\n",
                     getName(),
                     patchindex,
                     arborID,
                     (double)minL3NormTolerated);
               continue;
            }
            for (auto &weights : mWeightsList) {
               int nxp               = weights->getPatchSizeX();
               int nyp               = weights->getPatchSizeY();
               int nfp               = weights->getPatchSizeF();
               int weights_per_patch = nxp * nyp * nfp;
               float *dataStartPatch = weights0->getData(arborID) + patchindex * weights_per_patch;
               normalizePatch(dataStartPatch, weights_per_patch, scaleFactor / l3norm);
            }
         }
      }
   }
   else {
      for (int patchindex = 0; patchindex < numDataPatches; patchindex++) {
         float sumcubed = 0.0f;
         for (int arborID = 0; arborID < nArbors; arborID++) {
            for (auto &weights : mWeightsList) {
               int nxp               = weights->getPatchSizeX();
               int nyp               = weights->getPatchSizeY();
               int nfp               = weights->getPatchSizeF();
               int xPatchStride      = weights->getPatchStrideX();
               int yPatchStride      = weights->getPatchStrideY();
               int weights_per_patch = nxp * nyp * nfp;
               float *dataStartPatch = weights->getData(arborID) + patchindex * weights_per_patch;
               for (int k = 0; k < weights_per_patch; k++) {
                  float w = fabs(dataStartPatch[k]);
                  sumcubed += w * w * w;
               }
            }
         }
         float l3norm = powf(sumcubed, 1.0f / 3.0f);
         if (fabsf(sumcubed) <= minL3NormTolerated) {
            WarnLog().printf(
                  "NormalizeL3 \"%s\": sum of squares of weights in patch %d is within "
                  "minL3NormTolerated=%f of zero.  Weights in this patch unchanged.\n",
                  getName(),
                  patchindex,
                  (double)minL3NormTolerated);
            continue;
         }
         for (int arborID = 0; arborID < nArbors; arborID++) {
            for (auto &weights : mWeightsList) {
               int nxp               = weights->getPatchSizeX();
               int nyp               = weights->getPatchSizeY();
               int nfp               = weights->getPatchSizeF();
               int weights_per_patch = nxp * nyp * nfp;
               float *dataStartPatch = weights->getData(arborID) + patchindex * weights_per_patch;
               normalizePatch(dataStartPatch, weights_per_patch, scaleFactor / l3norm);
            }
         }
      }
   }
   return status;
}

NormalizeL3::~NormalizeL3() {}

} // namespace PV
