/*
 * InitOneToOneWeights.cpp
 *
 *  Created on: Sep 28, 2011
 *      Author: kpeterson
 */

#include "InitOneToOneWeights.hpp"
#include "InitOneToOneWeightsParams.hpp"

namespace PV {

InitOneToOneWeights::InitOneToOneWeights()
{
   initialize_base();
}

InitOneToOneWeights::~InitOneToOneWeights()
{
   // TODO Auto-generated destructor stub
}

int InitOneToOneWeights::initialize_base() {
   return PV_SUCCESS;
}

InitWeightsParams * InitOneToOneWeights::createNewWeightParams(HyPerConn * callingConn) {
   InitWeightsParams * tempPtr = new InitOneToOneWeightsParams(callingConn);
   return tempPtr;
}

int InitOneToOneWeights::calcWeights(/* PVPatch * patch */ pvdata_t * dataStart, int patchIndex, int arborId,
                                   InitWeightsParams *weightParams) {

   InitOneToOneWeightsParams *weightParamPtr = dynamic_cast<InitOneToOneWeightsParams*>(weightParams);


   if(weightParamPtr==NULL) {
      fprintf(stderr, "Failed to recast pointer to weightsParam!  Exiting...");
      exit(PV_FAILURE); // return 1;
   }


   weightParamPtr->calcOtherParams(patchIndex);

   const float iWeight = weightParamPtr->getInitWeight();

   return createOneToOneConnection(dataStart, patchIndex, iWeight, weightParamPtr);
   //subUnitWeights(patch, weightParamPtr);


}

int InitOneToOneWeights::createOneToOneConnection(/* PVPatch * patch */ pvdata_t * dataStart, int dataPatchIndex, float iWeight, InitWeightsParams * weightParamPtr) {
   // int numKernels = numDataPatches(0);
   //for( int k=0; k < numKernels; k++ ) {
   //int k=patchIndex;
   int k=weightParamPtr->getParentConn()->dataIndexToUnitCellIndex(dataPatchIndex);
   // PVPatch * kp = patch; //getKernelPatch(k);
   //assert(kp->nf == nfPatch_tmp);
   // assert(kp->nx == nxPatch_tmp);
   // assert(kp->ny == nyPatch_tmp);

   // pvdata_t * w = kp->data;

   const int nfp = weightParamPtr->getnfPatch_tmp();
   const int nxp = weightParamPtr->getnxPatch_tmp();
   const int nyp = weightParamPtr->getnyPatch_tmp();

   // const int nxp = kp->nx;
   // const int nyp = kp->ny;
   // const int nfp = weightParamPtr->getnfPatch_tmp(); // kp->nf;

   const int sxp = weightParamPtr->getsx_tmp(); // kp->sx;
   const int syp = weightParamPtr->getsy_tmp(); //kp->sy;
   const int sfp = weightParamPtr->getsf_tmp(); //kp->sf;

   // loop over all post-synaptic cells in patch
   for (int y = 0; y < nyp; y++) {
      for (int x = 0; x < nxp; x++) {
         for (int f = 0; f < nfp; f++) {
            if((x!=(int)(nxp/2))||(y!=(int)(nyp/2)))
               dataStart[x * sxp + y * syp + f * sfp] = 0;
            else
               dataStart[x * sxp + y * syp + f * sfp] = f==k ? iWeight : 0;
        }
      }
   }
   //}
   return PV_SUCCESS; // return 1;

}

} /* namespace PV */
