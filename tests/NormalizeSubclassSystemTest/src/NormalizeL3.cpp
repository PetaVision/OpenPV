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

   FatalIf(connectionList.empty(), "normalizeWeights called with connectionList empty.\n");

   // All connections in the group must have the same values of sharedWeights, numArbors, and
   // numDataPatches
   HyPerConn *conn0 = connectionList[0];

   float scale_factor = 1.0f;
   if (normalizeFromPostPerspective) {
      if (conn0->usingSharedWeights() == false) {
         Fatal().printf(
               "NormalizeL3 error for %s: normalizeFromPostPerspective is true but connection does "
               "not use shared weights.\n",
               conn0->getDescription_c());
      }
      scale_factor = ((float)conn0->postSynapticLayer()->getNumNeurons())
                     / ((float)conn0->preSynapticLayer()->getNumNeurons());
   }
   scale_factor *= strength;

   status = NormalizeMultiply::normalizeWeights(); // applies normalize_cutoff threshold and
   // rMinX,rMinY

   int nArbors        = conn0->numberOfAxonalArborLists();
   int numDataPatches = conn0->getNumDataPatches();
   if (normalizeArborsIndividually) {
      for (int arborID = 0; arborID < nArbors; arborID++) {
         for (int patchindex = 0; patchindex < numDataPatches; patchindex++) {
            float sumcubed = 0.0f;
            for (auto &conn : connectionList) {
               int nxp               = conn->xPatchSize();
               int nyp               = conn->yPatchSize();
               int nfp               = conn->fPatchSize();
               int xPatchStride      = conn->xPatchStride();
               int yPatchStride      = conn->yPatchStride();
               int weights_per_patch = nxp * nyp * nfp;
               float *dataStartPatch =
                     conn->get_wDataStart(arborID) + patchindex * weights_per_patch;
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
            for (auto &conn : connectionList) {
               int nxp               = conn->xPatchSize();
               int nyp               = conn->yPatchSize();
               int nfp               = conn->fPatchSize();
               int weights_per_patch = nxp * nyp * nfp;
               float *dataStartPatch =
                     conn0->get_wDataStart(arborID) + patchindex * weights_per_patch;
               normalizePatch(dataStartPatch, weights_per_patch, scale_factor / l3norm);
            }
         }
      }
   }
   else {
      for (int patchindex = 0; patchindex < numDataPatches; patchindex++) {
         float sumcubed = 0.0f;
         for (int arborID = 0; arborID < nArbors; arborID++) {
            for (auto &conn : connectionList) {
               int nxp               = conn->xPatchSize();
               int nyp               = conn->yPatchSize();
               int nfp               = conn->fPatchSize();
               int xPatchStride      = conn->xPatchStride();
               int yPatchStride      = conn->yPatchStride();
               int weights_per_patch = nxp * nyp * nfp;
               float *dataStartPatch =
                     conn->get_wDataStart(arborID) + patchindex * weights_per_patch;
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
            for (auto &conn : connectionList) {
               int nxp               = conn->xPatchSize();
               int nyp               = conn->yPatchSize();
               int nfp               = conn->fPatchSize();
               int weights_per_patch = nxp * nyp * nfp;
               float *dataStartPatch =
                     conn->get_wDataStart(arborID) + patchindex * weights_per_patch;
               normalizePatch(dataStartPatch, weights_per_patch, scale_factor / l3norm);
            }
         }
      }
   }
   return status;
}

NormalizeL3::~NormalizeL3() {}

} // namespace PV
