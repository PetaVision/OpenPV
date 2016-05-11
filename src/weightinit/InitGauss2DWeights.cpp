/*
 * InitGauss2DWeights.cpp
 *
 *  Created on: Apr 8, 2013
 *      Author: garkenyon
 */

#include "InitGauss2DWeights.hpp"

namespace PV {

InitGauss2DWeights::InitGauss2DWeights(char const * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

InitGauss2DWeights::InitGauss2DWeights() {
   initialize_base();
}

InitGauss2DWeights::~InitGauss2DWeights() {
}

int InitGauss2DWeights::initialize_base() {
   return PV_SUCCESS;
}

int InitGauss2DWeights::initialize(char const * name, HyPerCol * hc) {
   int status = InitWeights::initialize(name, hc);
   return status;
}

InitWeightsParams * InitGauss2DWeights::createNewWeightParams() {
   InitWeightsParams * tempPtr = new InitGauss2DWeightsParams(name, parent);
   return tempPtr;
}


int InitGauss2DWeights::calcWeights(pvdata_t * dataStart, int dataPatchIndex, int arborId) {

    InitGauss2DWeightsParams *weightParamPtr = dynamic_cast<InitGauss2DWeightsParams*> (weightParams);

    if(weightParamPtr==NULL) {
       fprintf(stderr, "Failed to recast pointer to weightsParam!  Exiting...");
       exit(PV_FAILURE);
    }

    weightParamPtr->calcOtherParams(dataPatchIndex);

    //calculate the weights:
    gauss2DCalcWeights(dataStart, weightParamPtr);

    return PV_SUCCESS;
}


/**
 * calculate gaussian weights between oriented line segments
 */
int InitGauss2DWeights::gauss2DCalcWeights(pvdata_t * dataStart, InitGauss2DWeightsParams * weightParamPtr) {

   //load necessary params:
   int nfPatch_tmp = weightParamPtr->getnfPatch();
   int nyPatch_tmp = weightParamPtr->getnyPatch();
   int nxPatch_tmp = weightParamPtr->getnxPatch();
   float strength=weightParamPtr->getStrength();
   float aspect=weightParamPtr->getAspect();
   float shift=weightParamPtr->getShift();
   int numFlanks=weightParamPtr->getNumFlanks();
   float sigma=weightParamPtr->getSigma();
   int sx_tmp=weightParamPtr->getsx();
   int sy_tmp=weightParamPtr->getsy();
   int sf_tmp=weightParamPtr->getsf();
   double r2Max=weightParamPtr->getr2Max();
   double r2Min=weightParamPtr->getr2Min();

   pvdata_t * w_tmp = dataStart;



   // loop over all post-synaptic cells in temporary patch
   for (int fPost = 0; fPost < nfPatch_tmp; fPost++) {
      float thPost = weightParamPtr->calcThPost(fPost);
      //TODO: add additional weight factor for difference between thPre and thPost
      if(weightParamPtr->checkThetaDiff(thPost)) continue;
      if(weightParamPtr->checkColorDiff(fPost)) continue;
      for (int jPost = 0; jPost < nyPatch_tmp; jPost++) {
         float yDelta = weightParamPtr->calcYDelta(jPost);
         for (int iPost = 0; iPost < nxPatch_tmp; iPost++) {
            float xDelta = weightParamPtr->calcXDelta(iPost);

            if(weightParamPtr->isSameLocOrSelf(xDelta, yDelta, fPost)) continue;

            // rotate the reference frame by th (change sign of thPost?)
            float xp = +xDelta * cos(thPost) + yDelta * sin(thPost);
            float yp = -xDelta * sin(thPost) + yDelta * cos(thPost);

            if(weightParamPtr->checkBowtieAngle(yp, xp)) continue;


            // include shift to flanks
            float d2 = xp * xp + (aspect * (yp - shift) * aspect * (yp - shift));
            int index = iPost * sx_tmp + jPost * sy_tmp + fPost * sf_tmp;
            w_tmp[index] = 0;
            if ((d2 <= r2Max) && (d2 >= r2Min)) {
               w_tmp[index] += strength*exp(-d2 / (2.0f * sigma * sigma));
            }
            if (numFlanks > 1) {
               // shift in opposite direction
               d2 = xp * xp + (aspect * (yp + shift) * aspect * (yp + shift));
               if ((d2 <= r2Max) && (d2 >= r2Min)) {
                  w_tmp[index] += strength*exp(-d2 / (2.0f * sigma * sigma));
               }
            }
         }
      }
   }

   return PV_SUCCESS;
}

BaseObject * createInitGauss2DWeights(char const * name, HyPerCol * hc) {
   return hc ? new InitGauss2DWeights(name, hc) : NULL;
}

} /* namespace PV */
