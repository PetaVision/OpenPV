/*
 * NormalizeSum.cpp
 *
 *  Created on: Apr 8, 2013
 *      Author: pschultz
 */

#include "NormalizeSum.hpp"
#include <iostream>

namespace PV {

NormalizeSum::NormalizeSum() {
   initialize_base();
}

NormalizeSum::NormalizeSum(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

int NormalizeSum::initialize_base() {
   return PV_SUCCESS;
}

int NormalizeSum::initialize(const char * name, HyPerCol * hc) {
   return NormalizeMultiply::initialize(name, hc);
}

int NormalizeSum::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = NormalizeMultiply::ioParamsFillGroup(ioFlag);
   ioParam_minSumTolerated(ioFlag);
   return status;
}

void NormalizeSum::ioParam_minSumTolerated(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "minSumTolerated", &minSumTolerated, 0.0f, true/*warnIfAbsent*/);
}

int NormalizeSum::normalizeWeights() {
   int status = PV_SUCCESS;

   assert(numConnections >= 1);

   // All connections in the group must have the same values of sharedWeights, numArbors, and numDataPatches
   HyPerConn * conn0 = connectionList[0];

   float scale_factor = 1.0f;
   if (normalizeFromPostPerspective) {
      if (conn0->usingSharedWeights()==false) {
         fprintf(stderr, "NormalizeSum error for connection \"%s\": normalizeFromPostPerspective is true but connection does not use shared weights.\n", getName());
         exit(EXIT_FAILURE);
      }
      scale_factor = ((float) conn0->postSynapticLayer()->getNumNeurons())/((float) conn0->preSynapticLayer()->getNumNeurons());
   }
   scale_factor *= strength;

   status = NormalizeBase::normalizeWeights(); // applies normalize_cutoff threshold and symmetrizeWeights


   int nArbors = conn0->numberOfAxonalArborLists();
   int numDataPatches = conn0->getNumDataPatches();
   if (normalizeArborsIndividually) {
	  for (int arborID = 0; arborID<nArbors; arborID++) {
		 for (int patchindex = 0; patchindex<numDataPatches; patchindex++) {
			double sum = 0.0;
			for (int c=0; c<numConnections; c++) {
			   HyPerConn * conn = connectionList[c];
               int nxp = conn->xPatchSize();
               int nyp = conn->yPatchSize();
               int nfp = conn->fPatchSize();
               int weights_per_patch = nxp*nyp*nfp;
			   pvwdata_t * dataStartPatch = conn->get_wDataHead(arborID,patchindex);
               accumulateSum(dataStartPatch, weights_per_patch, &sum);
			}
			if (fabs(sum) <= minSumTolerated) {
			   fprintf(stderr, "NormalizeSum warning for normalizer \"%s\": sum of weights in patch %d of arbor %d is within minSumTolerated=%f of zero. Weights in this patch unchanged.\n", getName(), patchindex, arborID, minSumTolerated);
			   break;
			}
            for (int c=0; c<numConnections; c++) {
               HyPerConn * conn = connectionList[c];
               int nxp = conn->xPatchSize();
               int nyp = conn->yPatchSize();
               int nfp = conn->fPatchSize();
               int weights_per_patch = nxp*nyp*nfp;
               pvwdata_t * dataStartPatch = conn->get_wDataHead(arborID,patchindex);
               normalizePatch(dataStartPatch, weights_per_patch, scale_factor/sum);
            }
		 }
	  }
   }
   else {
      for (int patchindex = 0; patchindex<numDataPatches; patchindex++) {
         double sum = 0.0;
         for (int arborID = 0; arborID<nArbors; arborID++) {
            for (int c=0; c<numConnections; c++) {
               HyPerConn * conn = connectionList[c];
               int nxp = conn->xPatchSize();
               int nyp = conn->yPatchSize();
               int nfp = conn->fPatchSize();
               int weights_per_patch = nxp*nyp*nfp;
               pvwdata_t * dataStartPatch = conn->get_wDataHead(arborID,patchindex);
               accumulateSum(dataStartPatch, weights_per_patch, &sum);
            }
         }
         if (fabs(sum) <= minSumTolerated) {
            fprintf(stderr, "NormalizeSum warning for connection \"%s\": sum of weights in patch %d is within minSumTolerated=%f of zero.  Weights in this patch unchanged.\n", getName(), patchindex, minSumTolerated);
            break;

         }
         for (int arborID = 0; arborID<nArbors; arborID++) {
            for (int c=0; c<numConnections; c++) {
               HyPerConn * conn = connectionList[c];
               int nxp = conn->xPatchSize();
               int nyp = conn->yPatchSize();
               int nfp = conn->fPatchSize();
               int weights_per_patch = nxp*nyp*nfp;
               pvwdata_t * dataStartPatch = conn->get_wDataHead(arborID,patchindex);
               normalizePatch(dataStartPatch, weights_per_patch, scale_factor/sum);
            }
         }
      } // patchindex
   } // normalizeArborsIndividually
   return status;
}

NormalizeSum::~NormalizeSum() {
}

BaseObject * createNormalizeSum(char const * name, HyPerCol * hc) {
   return hc ? new NormalizeSum(name, hc) : NULL;
}

} /* namespace PV */
