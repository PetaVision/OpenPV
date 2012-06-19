/*
 * InitDistributedWeights.cpp
 *
 *  Created on: Jun 18, 2012
 *      Author: bnowers
 */

#include "InitDistributedWeights.hpp"
#include "InitDistributedWeightsParams.hpp"

#include "../utils/pv_random.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

namespace PV {

InitDistributedWeights::InitDistributedWeights()
{
   initialize_base();
}

InitDistributedWeights::~InitDistributedWeights()
{
   // TODO Auto-generated destructor stub
}

int InitDistributedWeights::initialize_base() {
   return PV_SUCCESS;
}

InitWeightsParams * InitDistributedWeights::createNewWeightParams(HyPerConn * callingConn) {
   InitWeightsParams * tempPtr = new InitDistributedWeightsParams(callingConn);
   return tempPtr;
}

int InitDistributedWeights::calcWeights(/* PVPatch * wp */ pvdata_t * dataStart, int patchIndex, int arborId,
      InitWeightsParams *weightParams) {
   InitDistributedWeightsParams *weightParamPtr = dynamic_cast<InitDistributedWeightsParams*>(weightParams);

   if(weightParamPtr==NULL) {
      fprintf(stderr, "Failed to recast pointer to weightsParam!  Exiting...");
      exit(1);
   }

   const float wMinInit = weightParamPtr->getWMin();
   const float wMaxInit = weightParamPtr->getWMax();

   distributedWeights(dataStart, wMinInit, wMaxInit, weightParamPtr);
   return PV_SUCCESS; // return 1;
}

int* randmatrix(int lowerb, int upperb, int num){
   int* ptr = NULL;
   int i = 0;
   int indices[num];
   srand(time(NULL));

   for(i = 0; i < num; i++){
      indices[i] = (rand() % (upperb - lowerb)) + lowerb;
   }

   ptr = indices;
   return ptr;
}

/**
 * generate random weights for a patch from a uniform distribution
 * NOTES:
 *    - the pointer w already points to the patch head in the data structure
 *    - it only sets the weights to "real" neurons, not to neurons in the boundary
 *    layer. For example, if x are boundary neurons and o are real neurons,
 *    x x x x
 *    x o o o
 *    x o o o
 *    x o o o
 *
 *    for a 4x4 connection it sets the weights to the o neurons only.
 *    .
 */
int InitDistributedWeights::distributedWeights(/* PVPatch * wp */ pvdata_t * dataStart, float minwgt, float maxwgt, InitDistributedWeightsParams *weightParamPtr) {
      // changed variable names to avoid confusion with data members this->wMin and this->wMax
   // pvdata_t * w = wp->data;
   const int numNodes = weightParamPtr->getNumNodes();
   int i = 0;
   int * ptr = randmatrix(1, 65536, numNodes);
   int indices[numNodes];
   memcpy(indices, ptr, numNodes * sizeof(*ptr));
   const int nxp = weightParamPtr->getnxPatch_tmp(); // wp->nx;
   const int nyp = weightParamPtr->getnyPatch_tmp(); // wp->ny;
   const int nfp = weightParamPtr->getnfPatch_tmp(); //wp->nf;

   const int sxp = weightParamPtr->getsx_tmp(); //wp->sx;
   const int syp = weightParamPtr->getsy_tmp(); //wp->sy;
   const int sfp = weightParamPtr->getsf_tmp(); //wp->sf;

   double p;
   if( maxwgt <= minwgt ) {
      if( maxwgt < minwgt ) {
         fprintf(stderr, "Warning: distributedWeights maximum less than minimum.  Changing max = %f to min value of %f\n", maxwgt, minwgt);
         maxwgt = minwgt;
      }
      p = 0;
   }
   else {
       p = (maxwgt - minwgt) / pv_random_max();
   }

   // loop over all post-synaptic cells in patch
   for (int y = 0; y < nyp; y++) {
      for (int x = 0; x < nxp; x++) {
         for (int f = 0; f < nfp; f++) {
            dataStart[x * sxp + y * syp + f * sfp] = 0;/*minwgt + p * pv_random();*/
         }
      }
   }

   for(i = 0; i < numNodes; i++){
         dataStart[indices[i]] = 1;
   }

   return PV_SUCCESS;
}


} /* namespace PV */
