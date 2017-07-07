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

   assert(!connectionList.empty());

   // All connections in the group must have the same values of sharedWeights, numArbors, and
   // numDataPatches
   HyPerConn *conn0 = connectionList[0];

   float scale_factor = 1.0f;
   if (normalizeFromPostPerspective) {
      if (conn0->usingSharedWeights() == false) {
         Fatal().printf(
               "NormalizeMax error for %s: normalizeFromPostPerspective is true but connection "
               "does not use shared weights.\n",
               getDescription_c());
      }
      scale_factor = ((float)conn0->postSynapticLayer()->getNumNeurons())
                     / ((float)conn0->preSynapticLayer()->getNumNeurons());
   }
   scale_factor *= strength;

   status = NormalizeMultiply::normalizeWeights(); // applies normalize_cutoff threshold and
   // symmetrizeWeights

   int nArbors        = conn0->numberOfAxonalArborLists();
   int numDataPatches = conn0->getNumDataPatches();
   if (normalizeArborsIndividually) {
      for (int arborID = 0; arborID < nArbors; arborID++) {
         for (int patchindex = 0; patchindex < numDataPatches; patchindex++) {
            float max = 0.0f;
            for (auto &conn : connectionList) {
               int nxp               = conn->xPatchSize();
               int nyp               = conn->yPatchSize();
               int nfp               = conn->fPatchSize();
               int weights_per_patch = nxp * nyp * nfp;
               float *dataStartPatch = conn->get_wDataHead(arborID, patchindex);
               accumulateMax(dataStartPatch, weights_per_patch, &max);
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
            for (auto &conn : connectionList) {
               int nxp               = conn->xPatchSize();
               int nyp               = conn->yPatchSize();
               int nfp               = conn->fPatchSize();
               int weights_per_patch = nxp * nyp * nfp;
               float *dataStartPatch = conn->get_wDataHead(arborID, patchindex);
               normalizePatch(dataStartPatch, weights_per_patch, scale_factor / max);
            }
         }
      }
   }
   else {
      for (int patchindex = 0; patchindex < numDataPatches; patchindex++) {
         float max = 0.0;
         for (int arborID = 0; arborID < nArbors; arborID++) {
            for (auto &conn : connectionList) {
               int nxp               = conn->xPatchSize();
               int nyp               = conn->yPatchSize();
               int nfp               = conn->fPatchSize();
               int weights_per_patch = nxp * nyp * nfp;
               float *dataStartPatch = conn->get_wDataHead(arborID, patchindex);
               accumulateMax(dataStartPatch, weights_per_patch, &max);
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
            for (auto &conn : connectionList) {
               int nxp               = conn->xPatchSize();
               int nyp               = conn->yPatchSize();
               int nfp               = conn->fPatchSize();
               int weights_per_patch = nxp * nyp * nfp;
               float *dataStartPatch = conn->get_wDataHead(arborID, patchindex);
               normalizePatch(dataStartPatch, weights_per_patch, scale_factor / max);
            }
         }
      }
   }
   return status;
}

NormalizeMax::~NormalizeMax() {}

} /* namespace PV */
