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

NormalizeSum::NormalizeSum(const char * name, HyPerCol * hc, HyPerConn ** connectionList, int numConnections) {
   initialize_base();
   initialize(name, hc, connectionList, numConnections);
}

int NormalizeSum::initialize_base() {
   return PV_SUCCESS;
}

int NormalizeSum::initialize(const char * name, HyPerCol * hc, HyPerConn ** connectionList, int numConnections) {
   return NormalizeBase::initialize(name, hc, connectionList, numConnections);
}

int NormalizeSum::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = NormalizeBase::ioParamsFillGroup(ioFlag);
   ioParam_minSumTolerated(ioFlag);
   return status;
}

void NormalizeSum::ioParam_minSumTolerated(enum ParamsIOFlag ioFlag) {
   parent()->ioParamValue(ioFlag, name, "minSumTolerated", &minSumTolerated, 0.0f, true/*warnIfAbsent*/);
}

int NormalizeSum::normalizeWeights() {
   int status = PV_SUCCESS;

   assert(numConnections==1); // TODO: generalize for groups of connections
   HyPerConn * conn = connectionList[0];

#ifdef USE_SHMGET
#ifdef PV_USE_MPI
   if (conn->getShmgetFlag() && !conn->getShmgetOwner(0)) { // Assumes that all arbors are owned by the same process
      MPI_Barrier(conn->getParent()->icCommunicator()->communicator());
      return status;
   }
#endif // PV_USE_MPI
#endif // USE_SHMGET
   float scale_factor = 1.0f;
   if (normalizeFromPostPerspective) {
      if (conn->usingSharedWeights()==false) {
         fprintf(stderr, "NormalizeSum error for connection \"%s\": normalizeFromPostPerspective is true but connection does not use shared weights.\n", conn->getName());
         exit(EXIT_FAILURE);
      }
      scale_factor = ((float) conn->postSynapticLayer()->getNumNeurons())/((float) conn->preSynapticLayer()->getNumNeurons());
   }
   scale_factor *= strength;

   status = NormalizeBase::normalizeWeights(); // applies normalize_cutoff threshold and symmetrizeWeights

   int nxp = conn->xPatchSize();
   int nyp = conn->yPatchSize();
   int nfp = conn->fPatchSize();
   int nxpShrunken = conn->getNxpShrunken();
   int nypShrunken = conn->getNypShrunken();
   int offsetShrunken = conn->getOffsetShrunken();
   int xPatchStride = conn->xPatchStride();
   int yPatchStride = conn->yPatchStride();
   int weights_per_patch = nxp*nyp*nfp;
   int nArbors = conn->numberOfAxonalArborLists();
   int numDataPatches = conn->getNumDataPatches();
   if (normalizeArborsIndividually) {
	  for (int arborID = 0; arborID<nArbors; arborID++) {
		 for (int patchindex = 0; patchindex<numDataPatches; patchindex++) {
			 pvwdata_t * dataStartPatch = conn->get_wDataHead(arborID,patchindex);
			double sum = 0.0;
			if (offsetShrunken == 0){
				accumulateSum(dataStartPatch, weights_per_patch, &sum);
			}
			else{
				accumulateSumShrunken(dataStartPatch, &sum,
						nxpShrunken, nypShrunken, offsetShrunken, xPatchStride, yPatchStride);
			}
			if (fabs(sum) <= minSumTolerated) {
			   fprintf(stderr, "NormalizeSum warning for normalizer \"%s\": sum of weights in patch %d of arbor %d is within minSumTolerated=%f of zero. Weights in this patch unchanged.\n", conn->getName(), patchindex, arborID, minSumTolerated);
			   break;
			}
			normalizePatch(dataStartPatch, weights_per_patch, scale_factor/sum);
		 }
	  }
	  bool testNormalizationFlag = false;
	  if (testNormalizationFlag){
		  for (int arborID = 0; arborID<nArbors; arborID++) {
			 for (int patchindex = 0; patchindex<numDataPatches; patchindex++) {
				 pvwdata_t * dataStartPatch = conn->get_wDataHead(arborID,patchindex);
				double sum = 0.0;
				if (offsetShrunken == 0){
					accumulateSum(dataStartPatch, weights_per_patch, &sum);
				}
				else{
					accumulateSumShrunken(dataStartPatch, &sum,
							nxpShrunken, nypShrunken, offsetShrunken, xPatchStride, yPatchStride);
				}
			 }
		  }
	  } // testNormalizationFlag
   }
   else {
      for (int patchindex = 0; patchindex<numDataPatches; patchindex++) {
         double sum = 0.0;
         for (int arborID = 0; arborID<nArbors; arborID++) {
            pvwdata_t * dataStartPatch = conn->get_wDataHead(arborID,patchindex);
            if (offsetShrunken == 0){
            	accumulateSum(dataStartPatch, weights_per_patch, &sum);
            }
            else{
            	accumulateSumShrunken(dataStartPatch, &sum,
            			nxpShrunken, nypShrunken, offsetShrunken, xPatchStride, yPatchStride);
            }
         }
         if (fabs(sum) <= minSumTolerated) {
            fprintf(stderr, "NormalizeSum warning for connection \"%s\": sum of weights in patch %d is within minSumTolerated=%f of zero.  Weights in this patch unchanged.\n", conn->getName(), patchindex, minSumTolerated);
            break;

         }
         for (int arborID = 0; arborID<nArbors; arborID++) {
            pvwdata_t * dataStartPatch = conn->get_wDataHead(arborID,patchindex);
            normalizePatch(dataStartPatch, weights_per_patch, scale_factor/sum);
         }
      } // patchindex
      bool testNormalizationFlag = false;
      float tol = 1e-6;
      if (testNormalizationFlag){
		  for (int patchindex = 0; patchindex<numDataPatches; patchindex++) {
			 double sum = 0.0;
			 for (int arborID = 0; arborID<nArbors; arborID++) {
				 pvwdata_t * dataStartPatch = conn->get_wDataHead(arborID,patchindex);
				 if (offsetShrunken == 0){
					accumulateSum(dataStartPatch, weights_per_patch, &sum);
				 }
				 else{
					 accumulateSumShrunken(dataStartPatch, &sum,
							 nxpShrunken, nypShrunken, offsetShrunken, xPatchStride, yPatchStride);
				 }
			 }
			 if (sum > tol){
				 assert(sum <= scale_factor*(1+tol) && sum >= scale_factor*(1-tol));
			 }
			 else{
				 std::cout << conn->getName() << "::normalizeSum::sum < tol, sum = " << sum << ", tol = " << tol << std::endl;
			 }
		  } // patchindex
      } // testNormalizationFlag
   } // normalizeArborsIndividually
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

NormalizeSum::~NormalizeSum() {
}

} /* namespace PV */
