/*
 * InitSpreadOverArborsWeights.cpp
 *
 *  Created on: Sep 1, 2011
 *      Author: kpeterson
 */

#include "InitSpreadOverArborsWeights.hpp"
#include "InitSpreadOverArborsWeightsParams.hpp"

namespace PV {

InitSpreadOverArborsWeights::InitSpreadOverArborsWeights(char const * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

InitSpreadOverArborsWeights::InitSpreadOverArborsWeights()
{
   initialize_base();
}

InitSpreadOverArborsWeights::~InitSpreadOverArborsWeights()
{
}

int InitSpreadOverArborsWeights::initialize_base() {
   return PV_SUCCESS;
}

int InitSpreadOverArborsWeights::initialize(char const * name, HyPerCol * hc) {
   int status = InitGauss2DWeights::initialize(name, hc);
   return status;
}

InitWeightsParams * InitSpreadOverArborsWeights::createNewWeightParams() {
   InitWeightsParams * tempPtr = new InitSpreadOverArborsWeightsParams(name, parent);
   return tempPtr;
}

int InitSpreadOverArborsWeights::calcWeights(/* PVPatch * patch */ pvdata_t * dataStart, int patchIndex, int arborId) {
   InitSpreadOverArborsWeightsParams *weightParamPtr = dynamic_cast<InitSpreadOverArborsWeightsParams*>(weightParams);

   if(weightParamPtr==NULL) {
      fprintf(stderr, "Failed to recast pointer to weightsParam!  Exiting...");
      exit(1);
   }

   weightParamPtr->calcOtherParams(patchIndex);

   spreadOverArborsWeights(/* patch */ dataStart, arborId, weightParamPtr);

   return PV_SUCCESS; // return 1;
}

/**
 * Initializes all weights to iWeight
 *
 */
int InitSpreadOverArborsWeights::spreadOverArborsWeights(/* PVPatch * patch */ pvdata_t * dataStart, int arborId,
      InitSpreadOverArborsWeightsParams * weightParamPtr) {



   //load necessary params:
   int nfPatch_tmp = weightParamPtr->getnfPatch();
   int nyPatch_tmp = weightParamPtr->getnyPatch();
   int nxPatch_tmp = weightParamPtr->getnxPatch();
   int sx_tmp=weightParamPtr->getsx();
   int sy_tmp=weightParamPtr->getsy();
   int sf_tmp=weightParamPtr->getsf();

   const float iWeight = weightParamPtr->getInitWeight();
   const int nArbors = callingConn->numberOfAxonalArborLists();


   // pvdata_t * w_tmp = patch->data;

   // loop over all post-synaptic cells in temporary patch
   for (int fPost = 0; fPost < nfPatch_tmp; fPost++) {
      float thPost = weightParamPtr->calcThPost(fPost);
      if(weightParamPtr->checkThetaDiff(thPost)) continue;
      for (int jPost = 0; jPost < nyPatch_tmp; jPost++) {
         float yDelta = weightParamPtr->calcYDelta(jPost);
         for (int iPost = 0; iPost < nxPatch_tmp; iPost++) {
            float xDelta = weightParamPtr->calcXDelta(iPost);


            // rotate the reference frame by th (change sign of thPost?)
            float xp = +xDelta * cosf(thPost) + yDelta * sinf(thPost);
            float yp = -xDelta * sinf(thPost) + yDelta * cosf(thPost);


            float weight = 0;
            if (xp*xp+yp*yp<1e-4) {
               weight = iWeight/nArbors;
            }
            else {
               float theta2pi = atan2f(yp, xp)/(2*PI);
               unsigned int xpraw, ypraw, atanraw;
               union u { float f; unsigned int i;};
               union u f2u;
               f2u.f = xp; xpraw = f2u.i;
               f2u.f = yp; ypraw = f2u.i;
               f2u.f = theta2pi; atanraw = f2u.i;
               if(theta2pi<0) theta2pi+=1;
               if(theta2pi>=1) theta2pi-=1; // theta2pi should be in the range [0,1) but roundoff could make it exactly 1
               float zone = theta2pi*nArbors;

               float intpart;
               float fracpart = modff(zone, &intpart);
               assert(intpart>=0 && intpart<nArbors && fracpart>=0 && fracpart<1);

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

BaseObject * createInitSpreadOverArborsWeights(char const * name, HyPerCol * hc) {
   return hc ? new InitSpreadOverArborsWeights(name, hc) : NULL;
}

} /* namespace PV */
