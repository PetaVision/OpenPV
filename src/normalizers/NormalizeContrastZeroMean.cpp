/*
 * NormalizeContrastZeroMean.cpp
 *
 *  Created on: Apr 8, 2013
 *      Author: pschultz
 */

#include "NormalizeContrastZeroMean.hpp"

namespace PV {

NormalizeContrastZeroMean::NormalizeContrastZeroMean() {
   initialize_base();
}

NormalizeContrastZeroMean::NormalizeContrastZeroMean(HyPerConn * callingConn) {
   initialize(callingConn);
}

int NormalizeContrastZeroMean::initialize_base() {
   return PV_SUCCESS;
}

int NormalizeContrastZeroMean::initialize(HyPerConn * callingConn) {
   return NormalizeBase::initialize(callingConn);
}

int NormalizeContrastZeroMean::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = NormalizeBase::ioParamsFillGroup(ioFlag);
   ioParam_minSumTolerated(ioFlag);
   return status;
}

void NormalizeContrastZeroMean::ioParam_minSumTolerated(enum ParamsIOFlag ioFlag) {
   parent()->ioParamValue(ioFlag, name, "minSumTolerated", &minSumTolerated, 0.0f, true/*warnIfAbsent*/);
}

void NormalizeContrastZeroMean::ioParam_normalizeFromPostPerspective(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      if (parent()->parameters()->present(name, "normalizeFromPostPerspective")) {
         if (parent()->columnId()==0) {
            fprintf(stderr, "%s \"%s\": normalizeMethod \"normalizeContrastZeroMean\" doesn't use normalizeFromPostPerspective parameter.\n",
                  parent()->parameters()->groupKeywordFromName(name), name);
         }
         parent()->parameters()->value(name, "normalizeFromPostPerspective"); // marks param as having been read
      }
   }
}

int NormalizeContrastZeroMean::normalizeWeights(HyPerConn * conn) {
   int status = PV_SUCCESS;
#ifdef USE_SHMGET
#ifdef PV_USE_MPI
   if (conn->getShmgetFlag() && !conn->getShmgetOwner(0)) { // Assumes that all arbors are owned by the same process
      MPI_Barrier(conn->getParent()->icCommunicator()->communicator());
      return status;
   }
#endif // PV_USE_MPI
#endif // USE_SHMGET

   float scale_factor = strength;

   status = NormalizeBase::normalizeWeights(conn); // applies normalize_cutoff threshold and symmetrizeWeights

   int nxp = conn->xPatchSize();
   int nyp = conn->yPatchSize();
   int nfp = conn->fPatchSize();
   int weights_per_patch = nxp*nyp*nfp;
   int nArbors = conn->numberOfAxonalArborLists();
   int numDataPatches = conn->getNumDataPatches();
   if (normalizeArborsIndividually) {
      for (int arborID = 0; arborID<nArbors; arborID++) {
         for (int patchindex = 0; patchindex<numDataPatches; patchindex++) {
            pvwdata_t * dataStartPatch = conn->get_wDataStart(arborID) + patchindex * weights_per_patch;
            double sum = 0.0;
            double sumsq = 0.0;
            accumulateSumAndSumSquared(dataStartPatch, weights_per_patch, &sum, &sumsq);
            if (fabs(sum) <= minSumTolerated) {
               fprintf(stderr, "NormalizeContrastZeroMean warning for normalizer \"%s\": sum of weights in patch %d of arbor %d is within minSumTolerated=%f of zero. Weights in this patch unchanged.\n", conn->getName(), patchindex, arborID, minSumTolerated);
               break;
            }
            float mean = sum/weights_per_patch;
            float var = sumsq/weights_per_patch - mean*mean;
            subtractOffsetAndNormalize(dataStartPatch, weights_per_patch, sum/weights_per_patch, sqrt(var)/scale_factor);
         }
      }
   }
   else {
      for (int patchindex = 0; patchindex<numDataPatches; patchindex++) {
         double sum = 0.0;
         double sumsq = 0.0;
         for (int arborID = 0; arborID<nArbors; arborID++) {
            pvwdata_t * dataStartPatch = conn->get_wDataStart(arborID)+patchindex*weights_per_patch;
            accumulateSumAndSumSquared(dataStartPatch, weights_per_patch, &sum, &sumsq);
         }
         if (fabs(sum) <= minSumTolerated) {
            fprintf(stderr, "NormalizeSum warning for connection \"%s\": sum of weights in patch %d is within minSumTolerated=%f of zero. Weights in this patch unchanged.\n", conn->getName(), patchindex, minSumTolerated);
            break;
         }
         int count = weights_per_patch*nArbors;
         float mean = sum/count;
         float var = sumsq/count - mean*mean;
         for (int arborID = 0; arborID<nArbors; arborID++) {
            pvwdata_t * dataStartPatch = conn->get_wDataStart(arborID)+patchindex*weights_per_patch;
            subtractOffsetAndNormalize(dataStartPatch, weights_per_patch, mean, sqrt(var)/scale_factor);
         }
      }
   }
#ifdef USE_SHMGET
#ifdef PV_USE_MPI
   if (conn->getShmgetFlag()) {
      assert(conn->getShmgetOwner(0)); // Assumes that all arbors are owned by the same process
      MPI_Barrier(conn->getParent()->icCommunicator()->communicator());
   }
#endif // PV_USE_MPI
#endif // USE_SHMGET
   return status;
}

void NormalizeContrastZeroMean::subtractOffsetAndNormalize(pvdata_t * dataStartPatch, int weights_per_patch, float offset, float normalizer) {
   for (int k=0; k<weights_per_patch; k++) {
      dataStartPatch[k] -= offset;
      dataStartPatch[k] /= normalizer;
   }
}

int NormalizeContrastZeroMean::accumulateSumAndSumSquared(pvdata_t * dataPatchStart, int weights_in_patch, double * sum, double * sumsq) {
   // Do not call with sum uninitialized.
   // sum, sumsq, max are not cleared inside this routine so that you can accumulate the stats over several patches with multiple calls
   for (int k=0; k<weights_in_patch; k++) {
      pvdata_t w = dataPatchStart[k];
      *sum += w;
      *sumsq += w*w;
   }
   return PV_SUCCESS;
}

NormalizeContrastZeroMean::~NormalizeContrastZeroMean() {
}

} /* namespace PV */
