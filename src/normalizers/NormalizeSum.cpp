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
         ioFlag, name, "minSumTolerated", &minSumTolerated, 0.0f, true /*warnIfAbsent*/);
}

int NormalizeSum::normalizeWeights() {
   int status = PV_SUCCESS;

   pvAssert(!connectionList.empty());

   // All connections in the group must have the same values of sharedWeights, numArbors, and
   // numDataPatches
   HyPerConn *conn0 = connectionList[0];

   float scale_factor = 1.0f;
   if (normalizeFromPostPerspective) {
      if (conn0->usingSharedWeights() == false) {
         Fatal().printf(
               "NormalizeSum error for %s: normalizeFromPostPerspective is true but connection "
               "does not use shared weights.\n",
               getDescription_c());
      }
      scale_factor = ((float)conn0->postSynapticLayer()->getNumNeurons())
                     / ((float)conn0->preSynapticLayer()->getNumNeurons());
   }
   scale_factor *= strength;

   status = NormalizeBase::normalizeWeights(); // applies normalize_cutoff threshold and
   // symmetrizeWeights

   int nArbors        = conn0->numberOfAxonalArborLists();
   int numDataPatches = conn0->getNumDataPatches();
   if (normalizeArborsIndividually) {
      for (int arborID = 0; arborID < nArbors; arborID++) {
         for (int patchindex = 0; patchindex < numDataPatches; patchindex++) {
            float sum = 0.0;
            for (auto &conn : connectionList) {
               int nxp               = conn->xPatchSize();
               int nyp               = conn->yPatchSize();
               int nfp               = conn->fPatchSize();
               int weights_per_patch = nxp * nyp * nfp;
               float *dataStartPatch = conn->get_wDataHead(arborID, patchindex);
               accumulateSum(dataStartPatch, weights_per_patch, &sum);
            }
            if (fabsf(sum) <= minSumTolerated) {
               WarnLog().printf(
                     "NormalizeSum for %s: sum of weights in patch %d of arbor %d is within "
                     "minSumTolerated=%f of zero. Weights in this patch unchanged.\n",
                     getDescription_c(),
                     patchindex,
                     arborID,
                     (double)minSumTolerated);
               continue;
            }
            for (auto &conn : connectionList) {
               int nxp               = conn->xPatchSize();
               int nyp               = conn->yPatchSize();
               int nfp               = conn->fPatchSize();
               int weights_per_patch = nxp * nyp * nfp;
               float *dataStartPatch = conn->get_wDataHead(arborID, patchindex);
               normalizePatch(dataStartPatch, weights_per_patch, scale_factor / sum);
            }
         }
      }
   }
   else {
      for (int patchindex = 0; patchindex < numDataPatches; patchindex++) {
         float sum = 0.0;
         for (int arborID = 0; arborID < nArbors; arborID++) {
            for (auto &conn : connectionList) {
               int nxp               = conn->xPatchSize();
               int nyp               = conn->yPatchSize();
               int nfp               = conn->fPatchSize();
               int weights_per_patch = nxp * nyp * nfp;
               float *dataStartPatch = conn->get_wDataHead(arborID, patchindex);
               accumulateSum(dataStartPatch, weights_per_patch, &sum);
            }
         }
         if (fabsf(sum) <= minSumTolerated) {
            WarnLog().printf(
                  "NormalizeSum for %s: sum of weights in patch %d is within minSumTolerated=%f of "
                  "zero.  Weights in this patch unchanged.\n",
                  getDescription_c(),
                  patchindex,
                  (double)minSumTolerated);
            continue;
         }
         for (int arborID = 0; arborID < nArbors; arborID++) {
            for (auto &conn : connectionList) {
               int nxp               = conn->xPatchSize();
               int nyp               = conn->yPatchSize();
               int nfp               = conn->fPatchSize();
               int weights_per_patch = nxp * nyp * nfp;
               float *dataStartPatch = conn->get_wDataHead(arborID, patchindex);
               normalizePatch(dataStartPatch, weights_per_patch, scale_factor / sum);
            }
         }
      } // patchindex
   } // normalizeArborsIndividually
   return status;
}

NormalizeSum::~NormalizeSum() {}

} /* namespace PV */
