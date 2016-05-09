/*
 * InitDistributedWeights.cpp
 *
 *  Created on: Jun 18, 2012
 *      Author: bnowers
 *
 *  NOTES: This weight initialization class can ONLY be used in a HyPer Connection. It will
 *  not work with a Kernel Connection. The purpose of this class is to sparsely fill the patch
 *  matrix with a specified amount of neurons (nodes) that are randomly distributed throughout
 *  the matrix. To specify the number of nodes, add a numNodes parameter to the HyPerConn you
 *  wish to use in the params file.
 */

#include "InitDistributedWeights.hpp"
#include "InitDistributedWeightsParams.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

namespace PV {

InitDistributedWeights::InitDistributedWeights(HyPerConn * conn) {
   initialize_base();
   initialize(conn);
}

InitDistributedWeights::InitDistributedWeights()
{
   initialize_base();
}

InitDistributedWeights::~InitDistributedWeights()
{
}

int InitDistributedWeights::initialize_base() {
   return PV_SUCCESS;
}

int InitDistributedWeights::initialize(HyPerConn * conn) {
   int status = InitWeights::initialize(conn);
   return status;
}

InitWeightsParams * InitDistributedWeights::createNewWeightParams() {
   InitWeightsParams * tempPtr = new InitDistributedWeightsParams(callingConn);
   return tempPtr;
}

int InitDistributedWeights::calcWeights() {
   InitDistributedWeightsParams *weightParamPtr = dynamic_cast<InitDistributedWeightsParams*>(weightParams);
   const int numNodes = (int)weightParamPtr->getNumNodes(); //retrieves the number of nodes specified in the params file
   const int numDataPatches = callingConn->getNumDataPatches(); //retrieves the number of patches present in the current image
   int numArbors = callingConn->numberOfAxonalArborLists();
   assert(numArbors == 1); //makes sure #of arbors will always be 1
   assert(numNodes <= numDataPatches);
   int arbor = 0;
   int i = 0;
   int upperb = numDataPatches - 1;
   int lowerb = 0;
   const int nxp = weightParamPtr->getnxPatch();
   const int nyp = weightParamPtr->getnyPatch();
   const int nfp = weightParamPtr->getnfPatch();
   int patchSize = nfp*nxp*nyp;
   assert(patchSize == 1); //makes sure that nxp, nyp and nfp will always be 1

   int flag = 0;
   int max = 0;
   if(numNodes < (numDataPatches / 2)){
      flag = 1;
      max = numNodes;
   }
   else{
      flag = 0;
      max = numDataPatches - numNodes;
   }
   //the loop zeros out all the weights in the matrix
   for(int j = 0; j < numDataPatches; j++){
      pvwdata_t *weightptr = callingConn->get_wDataHead(arbor, j);
      if (flag) {
         *weightptr = 0;
      }
      else{
         *weightptr = 1;
      }
   }

   srand(time(NULL));
   //this loop receives a random index,then checks to see if it is a duplicate. If not, it sets the weight at that index to 1
   while(i < max) {
      int dataPatchIndex = (rand() % (upperb - lowerb)) + lowerb; //creates a random index in which to place a node based on the the upper and lower bounds
      pvwdata_t *weightptr = callingConn->get_wDataHead(arbor, dataPatchIndex);
      if (flag) {
         if(*weightptr == 0){
            *weightptr = 1;
            i++;
         }
      }
      else{
         if(*weightptr == 1){
            *weightptr = 0;
            i++;
         }
      }
   }
   return PV_SUCCESS;
}

} /* namespace PV */
