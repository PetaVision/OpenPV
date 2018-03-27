/*
 * NormalizeContrastZeroMean.cpp
 *
 *  Created on: Apr 8, 2013
 *      Author: pschultz
 */

#include "NormalizeContrastZeroMean.hpp"

namespace PV {

NormalizeContrastZeroMean::NormalizeContrastZeroMean() { initialize_base(); }

NormalizeContrastZeroMean::NormalizeContrastZeroMean(const char *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

int NormalizeContrastZeroMean::initialize_base() { return PV_SUCCESS; }

int NormalizeContrastZeroMean::initialize(const char *name, HyPerCol *hc) {
   return NormalizeBase::initialize(name, hc);
}

int NormalizeContrastZeroMean::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = NormalizeBase::ioParamsFillGroup(ioFlag);
   ioParam_minSumTolerated(ioFlag);
   return status;
}

void NormalizeContrastZeroMean::ioParam_minSumTolerated(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "minSumTolerated", &minSumTolerated, 0.0f, true /*warnIfAbsent*/);
}

void NormalizeContrastZeroMean::ioParam_normalizeFromPostPerspective(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      if (parent->parameters()->present(name, "normalizeFromPostPerspective")) {
         if (parent->columnId() == 0) {
            WarnLog().printf(
                  "%s \"%s\": normalizeMethod \"normalizeContrastZeroMean\" doesn't use "
                  "normalizeFromPostPerspective parameter.\n",
                  parent->parameters()->groupKeywordFromName(name),
                  name);
         }
         parent->parameters()->value(
               name, "normalizeFromPostPerspective"); // marks param as having been read
      }
   }
}

int NormalizeContrastZeroMean::normalizeWeights() {
   int status = PV_SUCCESS;

   pvAssert(!mWeightsList.empty());

   // TODO: need to ensure that all connections in mWeightsList have same
   // nxp,nyp,nfp,numArbors,numDataPatches
   Weights *weights0 = mWeightsList[0];
   for (auto &weights : mWeightsList) {
      if (weights->getNumArbors() != weights0->getNumArbors()) {
         if (parent->columnId() == 0) {
            ErrorLog().printf(
                  "%s: All connections in the normalization group must have the same number of "
                  "arbors (%s has %d; %s has %d).\n",
                  getDescription_c(),
                  weights0->getName().c_str(),
                  weights0->getNumArbors(),
                  weights->getName().c_str(),
                  weights->getNumArbors());
         }
         status = PV_FAILURE;
      }
      if (weights->getNumDataPatches() != weights0->getNumDataPatches()) {
         if (parent->columnId() == 0) {
            ErrorLog().printf(
                  "%s: All connections in the normalization group must have the same number of "
                  "data patches (%s has %d; %s has %d).\n",
                  getDescription_c(),
                  weights0->getName().c_str(),
                  weights0->getNumDataPatches(),
                  weights->getName().c_str(),
                  weights->getNumDataPatches());
         }
         status = PV_FAILURE;
      }
      if (status == PV_FAILURE) {
         MPI_Barrier(parent->getCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
   }

   float scale_factor = mStrength;

   status = NormalizeBase::normalizeWeights(); // applies normalize_cutoff threshold and
   // symmetrizeWeights

   int nArbors        = weights0->getNumArbors();
   int numDataPatches = weights0->getNumDataPatches();
   if (mNormalizeArborsIndividually) {
      for (int arborID = 0; arborID < nArbors; arborID++) {
         for (int patchindex = 0; patchindex < numDataPatches; patchindex++) {
            float sum           = 0.0f;
            float sumsq         = 0.0f;
            int weightsPerPatch = 0;
            for (auto &weights : mWeightsList) {
               int nxp = weights0->getPatchSizeX();
               int nyp = weights0->getPatchSizeY();
               int nfp = weights0->getPatchSizeF();
               weightsPerPatch += nxp * nyp * nfp;
               float *dataStartPatch = weights->getData(arborID) + patchindex * weightsPerPatch;
               accumulateSumAndSumSquared(dataStartPatch, weightsPerPatch, &sum, &sumsq);
            }
            if (fabsf(sum) <= minSumTolerated) {
               WarnLog().printf(
                     "for NormalizeContrastZeroMean \"%s\": sum of weights in patch %d of arbor %d "
                     "is within minSumTolerated=%f of zero. Weights in this patch unchanged.\n",
                     this->getName(),
                     patchindex,
                     arborID,
                     (double)minSumTolerated);
               continue;
            }
            float mean = sum / weightsPerPatch;
            float var  = sumsq / weightsPerPatch - mean * mean;
            for (auto &weights : mWeightsList) {
               float *dataStartPatch = weights->getData(arborID) + patchindex * weightsPerPatch;
               subtractOffsetAndNormalize(
                     dataStartPatch,
                     weightsPerPatch,
                     sum / weightsPerPatch,
                     sqrtf(var) / scale_factor);
            }
         }
      }
   }
   else {
      for (int patchindex = 0; patchindex < numDataPatches; patchindex++) {
         float sum           = 0.0f;
         float sumsq         = 0.0f;
         int weightsPerPatch = 0;
         for (int arborID = 0; arborID < nArbors; arborID++) {
            for (auto &weights : mWeightsList) {
               int nxp = weights0->getPatchSizeX();
               int nyp = weights0->getPatchSizeY();
               int nfp = weights0->getPatchSizeF();
               weightsPerPatch += nxp * nyp * nfp;
               float *dataStartPatch = weights->getData(arborID) + patchindex * weightsPerPatch;
               accumulateSumAndSumSquared(dataStartPatch, weightsPerPatch, &sum, &sumsq);
            }
         }
         if (fabsf(sum) <= minSumTolerated) {
            WarnLog().printf(
                  "for NormalizeContrastZeroMean \"%s\": sum of weights in patch %d is within "
                  "minSumTolerated=%f of zero. Weights in this patch unchanged.\n",
                  getName(),
                  patchindex,
                  (double)minSumTolerated);
            continue;
         }
         int count  = weightsPerPatch * nArbors;
         float mean = sum / count;
         float var  = sumsq / count - mean * mean;
         for (int arborID = 0; arborID < nArbors; arborID++) {
            for (auto &weights : mWeightsList) {
               float *dataStartPatch = weights->getData(arborID) + patchindex * weightsPerPatch;
               subtractOffsetAndNormalize(
                     dataStartPatch, weightsPerPatch, mean, sqrtf(var) / scale_factor);
            }
         }
      }
   }

   return status;
}

void NormalizeContrastZeroMean::subtractOffsetAndNormalize(
      float *dataStartPatch,
      int weightsPerPatch,
      float offset,
      float normalizer) {
   for (int k = 0; k < weightsPerPatch; k++) {
      dataStartPatch[k] -= offset;
      dataStartPatch[k] /= normalizer;
   }
}

int NormalizeContrastZeroMean::accumulateSumAndSumSquared(
      float *dataPatchStart,
      int weights_in_patch,
      float *sum,
      float *sumsq) {
   // Do not call with sum uninitialized.
   // sum, sumsq, max are not cleared inside this routine so that you can accumulate the stats over
   // several patches with multiple calls
   for (int k = 0; k < weights_in_patch; k++) {
      float w = dataPatchStart[k];
      *sum += w;
      *sumsq += w * w;
   }
   return PV_SUCCESS;
}

NormalizeContrastZeroMean::~NormalizeContrastZeroMean() {}

} /* namespace PV */
