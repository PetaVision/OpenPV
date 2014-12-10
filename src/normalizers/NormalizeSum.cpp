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
   return NormalizeMultiply::initialize(name, hc, connectionList, numConnections);
}

int NormalizeSum::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = NormalizeMultiply::ioParamsFillGroup(ioFlag);
   ioParam_minSumTolerated(ioFlag);
   return status;
}

void NormalizeSum::ioParam_minSumTolerated(enum ParamsIOFlag ioFlag) {
   parent()->ioParamValue(ioFlag, name, "minSumTolerated", &minSumTolerated, 0.0f, true/*warnIfAbsent*/);
}

int NormalizeSum::normalizeWeights() {
   int status = PV_SUCCESS;

   assert(numConnections >= 1);

   // TODO: need to ensure that all connections in connectionList have same sharedWeights,nxp,nyp,nfp,nxpShrunken,nypShrunken,offsetShrunken,sxp,syp,numArbors,numDataPatches,scale_factor
   HyPerConn * conn0 = connectionList[0];

#ifdef OBSOLETE // Marked obsolete Dec 9, 2014.
#ifdef USE_SHMGET
#ifdef PV_USE_MPI
   if (conn->getShmgetFlag() && !conn->getShmgetOwner(0)) { // Assumes that all arbors are owned by the same process
      MPI_Barrier(conn->getParent()->icCommunicator()->communicator());
      return status;
   }
#endif // PV_USE_MPI
#endif // USE_SHMGET
#endif // OBSOLETE
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

   int nxp = conn0->xPatchSize();
   int nyp = conn0->yPatchSize();
   int nfp = conn0->fPatchSize();
   int nxpShrunken = conn0->getNxpShrunken();
   int nypShrunken = conn0->getNypShrunken();
   int offsetShrunken = conn0->getOffsetShrunken();
   int xPatchStride = conn0->xPatchStride();
   int yPatchStride = conn0->yPatchStride();
   int weights_per_patch = nxp*nyp*nfp;
   int nArbors = conn0->numberOfAxonalArborLists();
   int numDataPatches = conn0->getNumDataPatches();
   if (normalizeArborsIndividually) {
	  for (int arborID = 0; arborID<nArbors; arborID++) {
		 for (int patchindex = 0; patchindex<numDataPatches; patchindex++) {
			double sum = 0.0;
			for (int c=0; c<numConnections; c++) {
			   HyPerConn * conn = connectionList[c];
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
			   fprintf(stderr, "NormalizeSum warning for normalizer \"%s\": sum of weights in patch %d of arbor %d is within minSumTolerated=%f of zero. Weights in this patch unchanged.\n", getName(), patchindex, arborID, minSumTolerated);
			   break;
			}
            for (int c=0; c<numConnections; c++) {
               HyPerConn * conn = connectionList[c];
               pvwdata_t * dataStartPatch = conn->get_wDataHead(arborID,patchindex);
               normalizePatch(dataStartPatch, weights_per_patch, scale_factor/sum);
            }
		 }
	  }
//	  bool testNormalizationFlag = false;
//	  if (testNormalizationFlag){
//		  for (int arborID = 0; arborID<nArbors; arborID++) {
//			 for (int patchindex = 0; patchindex<numDataPatches; patchindex++) {
//				double sum = 0.0;
//              pvwdata_t * dataStartPatch = conn->get_wDataHead(arborID,patchindex);
//				if (offsetShrunken == 0){
//					accumulateSum(dataStartPatch, weights_per_patch, &sum);
//				}
//				else{
//					accumulateSumShrunken(dataStartPatch, &sum,
//							nxpShrunken, nypShrunken, offsetShrunken, xPatchStride, yPatchStride);
//				}
//			 }
//		  }
//	  } // testNormalizationFlag
   }
   else {
      for (int patchindex = 0; patchindex<numDataPatches; patchindex++) {
         double sum = 0.0;
         for (int arborID = 0; arborID<nArbors; arborID++) {
            for (int c=0; c<numConnections; c++) {
               HyPerConn * conn = connectionList[c];
               pvwdata_t * dataStartPatch = conn->get_wDataHead(arborID,patchindex);
               if (offsetShrunken == 0){
                   accumulateSum(dataStartPatch, weights_per_patch, &sum);
               }
               else{
                   accumulateSumShrunken(dataStartPatch, &sum,
                           nxpShrunken, nypShrunken, offsetShrunken, xPatchStride, yPatchStride);
               }
            }
         }
         if (fabs(sum) <= minSumTolerated) {
            fprintf(stderr, "NormalizeSum warning for connection \"%s\": sum of weights in patch %d is within minSumTolerated=%f of zero.  Weights in this patch unchanged.\n", getName(), patchindex, minSumTolerated);
            break;

         }
         for (int arborID = 0; arborID<nArbors; arborID++) {
            for (int c=0; c<numConnections; c++) {
               HyPerConn * conn = connectionList[c];
               pvwdata_t * dataStartPatch = conn->get_wDataHead(arborID,patchindex);
               normalizePatch(dataStartPatch, weights_per_patch, scale_factor/sum);
            }
         }
      } // patchindex
//      bool testNormalizationFlag = false;
//      float tol = 1e-6;
//      if (testNormalizationFlag){
//		  for (int patchindex = 0; patchindex<numDataPatches; patchindex++) {
//			 double sum = 0.0;
//			 for (int arborID = 0; arborID<nArbors; arborID++) {
//				 pvwdata_t * dataStartPatch = conn->get_wDataHead(arborID,patchindex);
//				 if (offsetShrunken == 0){
//					accumulateSum(dataStartPatch, weights_per_patch, &sum);
//				 }
//				 else{
//					 accumulateSumShrunken(dataStartPatch, &sum,
//							 nxpShrunken, nypShrunken, offsetShrunken, xPatchStride, yPatchStride);
//				 }
//			 }
//			 if (sum > tol){
//				 assert(sum <= scale_factor*(1+tol) && sum >= scale_factor*(1-tol));
//			 }
//			 else{
//				 std::cout << conn->getName() << "::normalizeSum::sum < tol, sum = " << sum << ", tol = " << tol << std::endl;
//			 }
//		  } // patchindex
//      } // testNormalizationFlag
   } // normalizeArborsIndividually
#ifdef OBSOLETE // Marked obsolete Dec 9, 2014.
#ifdef USE_SHMGET
#ifdef PV_USE_MPI
   if (conn->getShmgetFlag()) {
      assert(conn->getShmgetOwner(0)); // Assumes that all arbors are owned by the same process
      MPI_Barrier(conn->getParent()->icCommunicator()->communicator());
   }
#endif // PV_USE_MPI
#endif // USE_SHMGET
#endif // OBSOLETE
   return status;
}

NormalizeSum::~NormalizeSum() {
}

} /* namespace PV */
