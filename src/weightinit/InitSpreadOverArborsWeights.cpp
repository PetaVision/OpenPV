/*
 * InitSpreadOverArborsWeights.cpp
 *
 *  Created on: Sep 1, 2011
 *      Author: kpeterson
 */

#include "InitSpreadOverArborsWeights.hpp"
#include "InitSpreadOverArborsWeightsParams.hpp"

namespace PV {

InitSpreadOverArborsWeights::InitSpreadOverArborsWeights()
{
   initialize_base();
}

InitSpreadOverArborsWeights::~InitSpreadOverArborsWeights()
{
   // TODO Auto-generated destructor stub
}

int InitSpreadOverArborsWeights::initialize_base() {
   return PV_SUCCESS;
}

InitWeightsParams * InitSpreadOverArborsWeights::createNewWeightParams(HyPerConn * callingConn) {
   InitWeightsParams * tempPtr = new InitSpreadOverArborsWeightsParams(callingConn);
   return tempPtr;
}

int InitSpreadOverArborsWeights::calcWeights(/* PVPatch * patch */ pvdata_t * dataStart, int patchIndex, int arborId,
      InitWeightsParams *weightParams) {
   InitSpreadOverArborsWeightsParams *weightParamPtr = dynamic_cast<InitSpreadOverArborsWeightsParams*>(weightParams);

   if(weightParamPtr==NULL) {
      fprintf(stderr, "Failed to recast pointer to weightsParam!  Exiting...");
      exit(1);
   }

   weightParamPtr->calcOtherParams(patchIndex);

   spreadOverArborsWeights(/* patch */ dataStart, arborId, weightParamPtr);
   weightParamPtr->getParentConn()->setDelay(arborId, arborId);
   return PV_SUCCESS; // return 1;
}

/**
 * Initializes all weights to iWeight
 *
 */
int InitSpreadOverArborsWeights::spreadOverArborsWeights(/* PVPatch * patch */ pvdata_t * dataStart, int arborId,
      InitSpreadOverArborsWeightsParams * weightParamPtr) {



   //load necessary params:
   int nfPatch_tmp = weightParamPtr->getnfPatch_tmp();
   int nyPatch_tmp = weightParamPtr->getnyPatch_tmp();
   int nxPatch_tmp = weightParamPtr->getnxPatch_tmp();
   int sx_tmp=weightParamPtr->getsx_tmp();
   int sy_tmp=weightParamPtr->getsy_tmp();
   int sf_tmp=weightParamPtr->getsf_tmp();

   const float iWeight = weightParamPtr->getInitWeight();
   const int nArbors = weightParamPtr->getNumArbors();


   // pvdata_t * w_tmp = patch->data;

   // loop over all post-synaptic cells in temporary patch
   for (int fPost = 0; fPost < nfPatch_tmp; fPost++) {
      float thPost = weightParamPtr->calcThPost(fPost);
      if(weightParamPtr->checkTheta(thPost)) continue;
      for (int jPost = 0; jPost < nyPatch_tmp; jPost++) {
         float yDelta = weightParamPtr->calcYDelta(jPost);
         for (int iPost = 0; iPost < nxPatch_tmp; iPost++) {
            float xDelta = weightParamPtr->calcXDelta(iPost);


            // rotate the reference frame by th (change sign of thPost?)
            float xp = +xDelta * cosf(thPost) + yDelta * sinf(thPost);
            float yp = -xDelta * sinf(thPost) + yDelta * cosf(thPost);

            if (fPost == 0 && jPost == 0 && iPost == 3) {
               printf("...\n");
            }



            float weight = 0;
            if (xp*xp+yp*yp<1e-4) {
               weight = iWeight/nArbors; // arborId ? 0 : iWeight;
            }
            else {
               float theta = atan2f(yp, xp);
               if(theta<0) theta+=2.0f*PI;
               float zone = theta/(2*PI)*nArbors;

               float intpart;
               float fracpart = modff(zone, &intpart);

               if (intpart==arborId) {
                  weight = iWeight*(1-fracpart);
               }
               else if ( (int) (intpart-arborId+1) % nArbors == 0) {
                  weight = iWeight*fracpart;
               }
            }
            // if (arborId == zone)
            //    weight = iWeight;

            int index = iPost * sx_tmp + jPost * sy_tmp + fPost * sf_tmp;
            dataStart[index] = weight;
         }
      }
   }

   return PV_SUCCESS;
}

} /* namespace PV */
