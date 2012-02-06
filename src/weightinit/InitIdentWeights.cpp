/*
 * InitIdentWeights.cpp
 *
 *  Created on: Aug 14, 2011
 *      Author: kpeterson
 */

#include "InitIdentWeights.hpp"
#include "InitIdentWeightsParams.hpp"

namespace PV {

InitIdentWeights::InitIdentWeights()
{
   initialize_base();
}

InitIdentWeights::~InitIdentWeights()
{
   // TODO Auto-generated destructor stub
}

int InitIdentWeights::initialize_base() {
   return PV_SUCCESS;
}

InitWeightsParams * InitIdentWeights::createNewWeightParams(HyPerConn * callingConn) {
   InitWeightsParams * tempPtr = new InitIdentWeightsParams(callingConn);
   return tempPtr;
}

int InitIdentWeights::calcWeights(PVPatch * patch, int patchIndex, int arborId,
                                   InitWeightsParams *weightParams) {

   InitIdentWeightsParams *weightParamPtr = dynamic_cast<InitIdentWeightsParams*>(weightParams);


   if(weightParamPtr==NULL) {
      fprintf(stderr, "Failed to recast pointer to weightsParam!  Exiting...");
      exit(PV_FAILURE); // return 1;
   }


   weightParamPtr->calcOtherParams(patch, patchIndex);

   //subUnitWeights(patch, weightParamPtr);

   //int numKernels = numDataPatches(0);
//   int nfPatch_tmp = weightParamPtr->getnfPatch_tmp();
//   //for( int k=0; k < numKernels; k++ ) {
//   //int k=patchIndex;
//   int k=weightParamPtr->getParentConn()->patchIndexToKernelIndex(patchIndex);
//   PVPatch * kp = patch; //getKernelPatch(k);
//   assert(kp->nf == nfPatch_tmp);
//   for( int l=0; l < kp->nf; l++ ) {
//        kp->data[l] = l==k;
//   }
   //}
   return createOneToOneConnection(patch, patchIndex, 1, weightParamPtr);
   //return PV_SUCCESS; // return 1;

}
} /* namespace PV */
