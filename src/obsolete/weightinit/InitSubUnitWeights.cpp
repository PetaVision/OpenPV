/*
 * InitSubUnitWeights.cpp
 *
 *  Created on: Aug 14, 2011
 *      Author: kpeterson
 */

#include "InitSubUnitWeights.hpp"
#include "InitSubUnitWeightsParams.hpp"

namespace PV {

InitSubUnitWeights::InitSubUnitWeights(HyPerConn * conn)
{
   initialize_base();
   initialize(conn);
}

InitSubUnitWeights::InitSubUnitWeights()
{
   initialize_base();
}

InitSubUnitWeights::~InitSubUnitWeights()
{
}

int InitSubUnitWeights::initialize_base() {
   return PV_SUCCESS;
}

int InitSubUnitWeights::initialize(HyPerConn * conn) {
   int status = InitWeights::initialize(conn);
   return status;
}

InitWeightsParams * InitSubUnitWeights::createNewWeightParams() {
   InitWeightsParams * tempPtr = new InitSubUnitWeightsParams(callingConn);
   return tempPtr;
}

int InitSubUnitWeights::calcWeights(/* PVPatch * patch */ pvdata_t * dataStart, int patchIndex, int arborId) {

   InitSubUnitWeightsParams *weightParamPtr = dynamic_cast<InitSubUnitWeightsParams*>(weightParams);


   if(weightParamPtr==NULL) {
      fprintf(stderr, "Failed to recast pointer to weightsParam!  Exiting...");
      exit(1);
   }


   weightParamPtr->calcOtherParams(patchIndex);

   subUnitWeights(dataStart, weightParamPtr);

   return PV_SUCCESS; // return 1;

}

/**
 * This connection is for retina to layer with 4 x 16 features.  The post-synaptic layer
 * exhaustively computes presence of a hierarchy of 4 x 2x2 (on/off) patch of pixels
 * (embedded in a 3x3 pixel patch).
 */
int InitSubUnitWeights::subUnitWeights(/* PVPatch * patch */ pvdata_t * dataStart, InitSubUnitWeightsParams * weightParamPtr) {
   assert(weightParamPtr->getPost()->clayer->loc.nf == 4*16);

   int nfPatch_tmp = weightParamPtr->getnfPatch();
   int nyPatch_tmp = weightParamPtr->getnyPatch();
   int nxPatch_tmp = weightParamPtr->getnxPatch();
   int sx_tmp=weightParamPtr->getsx();
   int sy_tmp=weightParamPtr->getsy();
   int sf_tmp=weightParamPtr->getsf();

   // pvdata_t * w_tmp = patch->data;

   // TODO - already initialized to zero (so delete)
   for (int k = 0; k < nxPatch_tmp*nyPatch_tmp*nfPatch_tmp; k++) {
      dataStart[k] = 0.0;
   }

   for (int f = 0; f < nfPatch_tmp; f++) {
      int i0 = 0, j0 = 0;
      int kf = f / 16;
      if (kf == 0) {i0 = 0; j0 = 0;}
      if (kf == 1) {i0 = 1; j0 = 0;}
      if (kf == 2) {i0 = 0; j0 = 1;}
      if (kf == 3) {i0 = 1; j0 = 1;}

      kf = f % 16;

      for (int j = 0; j < 2; j++) {
         for (int i = 0; i < 2; i++) {
            int n = i + 2*j;
            int r = kf >> n;
            r = 0x1 & r;
            dataStart[(i+i0)*sx_tmp + (j+j0)*sy_tmp + f*sf_tmp] = r;
         }
      }
   }

   // normalize
   for (int f = 0; f < nfPatch_tmp; f++) {
      float sum = 0;
      for (int i = 0; i < nxPatch_tmp*nyPatch_tmp; i++) sum += dataStart[f + i*nfPatch_tmp];

      if (sum == 0) continue;

      float factor = 1.0/sum;
      for (int i = 0; i < nxPatch_tmp*nyPatch_tmp; i++) dataStart[f + i*nfPatch_tmp] *= factor;
   }

   return 0;
}

} /* namespace PV */
