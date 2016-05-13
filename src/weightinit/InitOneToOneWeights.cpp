/*
 * InitOneToOneWeights.cpp
 *
 *  Created on: Sep 28, 2011
 *      Author: kpeterson
 */

#include "InitOneToOneWeights.hpp"
#include "InitOneToOneWeightsParams.hpp"

namespace PV {

InitOneToOneWeights::InitOneToOneWeights(char const * name, HyPerCol * hc)
{
   initialize_base();
   initialize(name, hc);
}

InitOneToOneWeights::InitOneToOneWeights()
{
   initialize_base();
}

InitOneToOneWeights::~InitOneToOneWeights()
{
}

int InitOneToOneWeights::initialize_base() {
   return PV_SUCCESS;
}

int InitOneToOneWeights::initialize(char const * name, HyPerCol * hc) {
   int status = InitWeights::initialize(name, hc);
   return status;
}

InitWeightsParams * InitOneToOneWeights::createNewWeightParams() {
   InitWeightsParams * tempPtr = new InitOneToOneWeightsParams(name, parent);
   return tempPtr;
}

int InitOneToOneWeights::calcWeights(pvdata_t * dataStart, int patchIndex, int arborId) {

   InitOneToOneWeightsParams *weightParamPtr = dynamic_cast<InitOneToOneWeightsParams*>(weightParams);


   if(weightParamPtr==NULL) {
      fprintf(stderr, "Failed to recast pointer to weightsParam!  Exiting...");
      exit(PV_FAILURE);
   }


   weightParamPtr->calcOtherParams(patchIndex);

   const float iWeight = weightParamPtr->getInitWeight();

   return createOneToOneConnection(dataStart, patchIndex, iWeight, weightParamPtr);
}

int InitOneToOneWeights::createOneToOneConnection(pvdata_t * dataStart, int dataPatchIndex, float iWeight, InitWeightsParams * weightParamPtr) {

   int k=weightParamPtr->getParentConn()->dataIndexToUnitCellIndex(dataPatchIndex);

   const int nfp = weightParamPtr->getnfPatch();
   const int nxp = weightParamPtr->getnxPatch();
   const int nyp = weightParamPtr->getnyPatch();

   const int sxp = weightParamPtr->getsx();
   const int syp = weightParamPtr->getsy();
   const int sfp = weightParamPtr->getsf();

   // clear all weights in patch
   memset(dataStart, 0, nxp*nyp*nfp);
   // then set the center point of the patch for each feature
   int x = (int) (nxp/2);
   int y = (int) (nyp/2);
   for (int f=0; f < nfp; f++) {
      dataStart[x * sxp + y * syp + f * sfp] = f == k ? iWeight : 0;
   }

   return PV_SUCCESS;

}

BaseObject * createInitOneToOneWeights(char const * name, HyPerCol * hc) {
   return hc ? new InitOneToOneWeights(name, hc) : NULL;
}

} /* namespace PV */
