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

NormalizeContrastZeroMean::NormalizeContrastZeroMean(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

int NormalizeContrastZeroMean::initialize_base() {
   return PV_SUCCESS;
}

int NormalizeContrastZeroMean::initialize(const char * name, HyPerCol * hc) {
   return NormalizeBase::initialize(name, hc);
}

int NormalizeContrastZeroMean::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = NormalizeBase::ioParamsFillGroup(ioFlag);
   ioParam_minSumTolerated(ioFlag);
   return status;
}

void NormalizeContrastZeroMean::ioParam_minSumTolerated(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "minSumTolerated", &minSumTolerated, 0.0f, true/*warnIfAbsent*/);
}

void NormalizeContrastZeroMean::ioParam_normalizeFromPostPerspective(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      if (parent->parameters()->present(name, "normalizeFromPostPerspective")) {
         if (parent->columnId()==0) {
            fprintf(stderr, "%s \"%s\": normalizeMethod \"normalizeContrastZeroMean\" doesn't use normalizeFromPostPerspective parameter.\n",
                  parent->parameters()->groupKeywordFromName(name), name);
         }
         parent->parameters()->value(name, "normalizeFromPostPerspective"); // marks param as having been read
      }
   }
}

int NormalizeContrastZeroMean::normalizeWeights() {
   int status = PV_SUCCESS;

   assert(numConnections >= 1);

   // TODO: need to ensure that all connections in connectionList have same nxp,nyp,nfp,numArbors,numDataPatches
   HyPerConn * conn0 = connectionList[0];
   for (int c=1; c<numConnections; c++) {
      HyPerConn * conn = connectionList[c];
      if (conn->numberOfAxonalArborLists() != conn0->numberOfAxonalArborLists()) {
         if (parent->columnId() == 0) {
            fprintf(stderr, "Normalizer %s: All connections in the normalization group must have the same number of arbors (Connection \"%s\" has %d; connection \"%s\" has %d).\n",
                  this->getName(), conn0->getName(), conn0->numberOfAxonalArborLists(), conn->getName(), conn->numberOfAxonalArborLists());
         }
         status = PV_FAILURE;
      }
      if (conn->getNumDataPatches() != conn0->getNumDataPatches()) {
         if (parent->columnId() == 0) {
            fprintf(stderr, "Normalizer %s: All connections in the normalization group must have the same number of data patches (Connection \"%s\" has %d; connection \"%s\" has %d).\n",
                  this->getName(), conn0->getName(), conn0->getNumDataPatches(), conn->getName(), conn->getNumDataPatches());
         }
         status = PV_FAILURE;
      }
      if (status==PV_FAILURE) {
         MPI_Barrier(parent->icCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
   }

   float scale_factor = strength;

   status = NormalizeBase::normalizeWeights(); // applies normalize_cutoff threshold and symmetrizeWeights

   int nArbors = conn0->numberOfAxonalArborLists();
   int numDataPatches = conn0->getNumDataPatches();
   if (normalizeArborsIndividually) {
      for (int arborID = 0; arborID<nArbors; arborID++) {
         for (int patchindex = 0; patchindex<numDataPatches; patchindex++) {
            double sum = 0.0;
            double sumsq = 0.0;
            int weights_per_patch = 0;
            for (int c=0; c<numConnections; c++) {
               HyPerConn * conn = connectionList[c];
               int nxp = conn0->xPatchSize();
               int nyp = conn0->yPatchSize();
               int nfp = conn0->fPatchSize();
               weights_per_patch += nxp*nyp*nfp;
               pvwdata_t * dataStartPatch = conn->get_wDataStart(arborID) + patchindex * weights_per_patch;
               accumulateSumAndSumSquared(dataStartPatch, weights_per_patch, &sum, &sumsq);
            }
            if (fabs(sum) <= minSumTolerated) {
               fprintf(stderr, "Warning for NormalizeContrastZeroMean \"%s\": sum of weights in patch %d of arbor %d is within minSumTolerated=%f of zero. Weights in this patch unchanged.\n", this->getName(), patchindex, arborID, minSumTolerated);
               break; // TODO: continue instead of break?  continue as opposed to break is more consistent with warning above.
            }
            float mean = sum/weights_per_patch;
            float var = sumsq/weights_per_patch - mean*mean;
            for (int c=0; c<numConnections; c++) {
               HyPerConn * conn = connectionList[c];
               pvwdata_t * dataStartPatch = conn->get_wDataStart(arborID) + patchindex * weights_per_patch;
               subtractOffsetAndNormalize(dataStartPatch, weights_per_patch, sum/weights_per_patch, sqrt(var)/scale_factor);
            }
         }
      }
   }
   else {
      for (int patchindex = 0; patchindex<numDataPatches; patchindex++) {
         double sum = 0.0;
         double sumsq = 0.0;
         int weights_per_patch = 0;
         for (int arborID = 0; arborID<nArbors; arborID++) {
            for (int c=0; c<numConnections; c++) {
               HyPerConn * conn = connectionList[c];
               int nxp = conn0->xPatchSize();
               int nyp = conn0->yPatchSize();
               int nfp = conn0->fPatchSize();
               weights_per_patch += nxp*nyp*nfp;
               pvwdata_t * dataStartPatch = conn->get_wDataStart(arborID)+patchindex*weights_per_patch;
               accumulateSumAndSumSquared(dataStartPatch, weights_per_patch, &sum, &sumsq);
            }
         }
         if (fabs(sum) <= minSumTolerated) {
            fprintf(stderr, "Warning for NormalizeContrastZeroMean \"%s\": sum of weights in patch %d is within minSumTolerated=%f of zero. Weights in this patch unchanged.\n", getName(), patchindex, minSumTolerated);
            break; // TODO: continue instead of break?  continue as opposed to break is more consistent with warning above.

         }
         int count = weights_per_patch*nArbors;
         float mean = sum/count;
         float var = sumsq/count - mean*mean;
         for (int arborID = 0; arborID<nArbors; arborID++) {
            for (int c=0; c<numConnections; c++) {
               HyPerConn * conn = connectionList[c];
               pvwdata_t * dataStartPatch = conn->get_wDataStart(arborID)+patchindex*weights_per_patch;
               subtractOffsetAndNormalize(dataStartPatch, weights_per_patch, mean, sqrt(var)/scale_factor);
            }
         }
      }
   }

   return status;
}

void NormalizeContrastZeroMean::subtractOffsetAndNormalize(pvwdata_t * dataStartPatch, int weights_per_patch, float offset, float normalizer) {
   for (int k=0; k<weights_per_patch; k++) {
      dataStartPatch[k] -= offset;
      dataStartPatch[k] /= normalizer;
   }
}

int NormalizeContrastZeroMean::accumulateSumAndSumSquared(pvwdata_t * dataPatchStart, int weights_in_patch, double * sum, double * sumsq) {
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

BaseObject * createNormalizeContrastZeroMean(char const * name, HyPerCol * hc) {
   return hc ? new NormalizeContrastZeroMean(name, hc) : NULL;
}

} /* namespace PV */
