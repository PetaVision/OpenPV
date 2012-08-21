/*
 * InitWeights.cpp
 *
 *  Created on: Aug 5, 2011
 *      Author: kpeterson
 */

#include "InitWeights.hpp"

#include <stdlib.h>

#include "../include/default_params.h"
#ifdef OBSOLETE // Marked obsolete Feb. 27, 2012.  Replaced by PatchProbe.
#include "../io/ConnectionProbe.hpp"
#endif // OBSOLETE
#include "../io/io.h"
#include "../io/fileio.hpp"
#include "../utils/conversions.h"
#include "../utils/pv_random.h"
#include "../columns/InterColComm.hpp"
#include "InitBIDSWeightsParams.hpp"
#include "InitBIDSWeights.hpp"

namespace PV {

InitBIDSWeights::InitBIDSWeights(){
   initialize_base();
}

InitBIDSWeights::~InitBIDSWeights(){
   // TODO Auto-generated destructor stub
}

/*This method does the three steps involved in initializing weights.  Subclasses shouldn't touch this method.
 * Subclasses should only generate their own versions of calcWeights to do their own type of weight initialization.
 *
 * This method initializes the full unshrunken patch.  The input argument numPatches is ignored.  Instead, method uses getNumDataPatches to determine number of
 * data patches.
 */
int InitBIDSWeights::initializeWeights(PVPatch *** patches, pvdata_t ** dataStart, int numPatches, const char * filename, HyPerConn * callingConn, float * timef /*default NULL*/){
   PVParams * inputParams = callingConn->getParent()->parameters();
   int initFromLastFlag = inputParams->value(callingConn->getName(), "initFromLastFlag", 0.0f, false) != 0;
   InitWeightsParams *weightParams = NULL;
   int numArbors = callingConn->numberOfAxonalArborLists();
   if( initFromLastFlag ) {
      char nametmp[PV_PATH_MAX];
      snprintf(nametmp, PV_PATH_MAX-1, "%s/w%1.1d_last.pvp", callingConn->getParent()->getOutputPath(), callingConn->getConnectionId());
      readWeights(patches, dataStart, callingConn->getNumDataPatches(), nametmp, callingConn);
   }
   else if( filename != NULL ) {
      readWeights(patches, dataStart, callingConn->getNumDataPatches(), filename, callingConn, timef);
   }
   else {
      weightParams = createNewWeightParams(callingConn);
      //int patchSize = nfp*nxp*nyp;

      for( int arbor=0; arbor<numArbors; arbor++ ) {
         for (int dataPatchIndex = 0; dataPatchIndex < callingConn->getNumDataPatches(); dataPatchIndex++) {

            //int correctedPatchIndex = callingConn->correctPIndex(patchIndex);
            //int correctedPatchIndex = patchIndex;
            //create full sized patch:
            // PVPatch * wp_tmp = createUnShrunkenPatch(callingConn, patches[arbor][patchIndex]);
            // if(wp_tmp==NULL) continue;

            //calc weights for patch:

            //int successFlag = calcWeights(get_wDataHead[arbor]+patchIndex*patchSize, patchIndex /*correctedPatchIndex*/, arbor, weightParams);
            //int successFlag = calcWeights(callingConn->get_wDataHead(arbor, patchIndex), patchIndex /*correctedPatchIndex*/, arbor, weightParams);
            int successFlag = calcWeightsBIDS(callingConn->get_wDataHead(arbor, dataPatchIndex), dataPatchIndex, arbor, weightParams, callingConn);
            if (successFlag != PV_SUCCESS) {
               fprintf(stderr, "Failed to create weights for %s! Exiting...\n", callingConn->getName());
               exit(PV_FAILURE);
            }

            //copy back to unshrunk patch:
            //copyToOriginalPatch(patches[arbor][patchIndex], wp_tmp);
            //copyToOriginalPatch(patches[arbor][patchIndex], wp_tmp,
            //      callingConn->get_wDataStart(arbor), patchIndex,
            //      callingConn->fPatchSize(), callingConn->yPatchStride());
            //free(wp_tmp);
         }
      }
      delete(weightParams);
   }
   //return callingConn->weights();
   return PV_SUCCESS;
}

InitWeightsParams * InitBIDSWeights::createNewWeightParams(HyPerConn * callingConn){
   InitWeightsParams * tempPtr = new InitBIDSWeightsParams(callingConn);
   return tempPtr;
}

int InitBIDSWeights::calcWeightsBIDS(/* PVPatch * patch */ pvdata_t * dataStart, int dataPatchIndex, int arborId, InitWeightsParams *weightParams, HyPerConn * conn){
    InitBIDSWeightsParams *weightParamPtr = dynamic_cast<InitBIDSWeightsParams*> (weightParams);

    if(weightParamPtr==NULL) {
       fprintf(stderr, "Failed to recast pointer to weightsParam!  Exiting...");
       exit(PV_FAILURE);
    }


    weightParamPtr->calcOtherParams(dataPatchIndex);

    //calculate the weights:
    BIDSCalcWeights(dataStart, weightParamPtr, conn);


    return PV_SUCCESS;
}

int InitBIDSWeights::initialize_base() {
   return PV_SUCCESS;
}


/**
 * calculate gaussian weights between oriented line segments
 */
int InitBIDSWeights::BIDSCalcWeights(/* PVPatch * patch */ pvdata_t * dataStart, InitBIDSWeightsParams * weightParamPtr, HyPerConn * conn) {

   //load necessary params:
   BIDSCoords * coords = weightParamPtr->getCoords();
   int numNodes = weightParamPtr->getNumNodes();

   // loop over all post-synaptic cells in temporary patch
   float maxDistance = (sqrt((256*256) + (256*256))); // / 10;
   for(int j = 0; j < weightParamPtr->getParentConn()->getNumDataPatches(); j++){
      for (int jPost = 0; jPost < numNodes; jPost++) {
         float yDelta = coords[jPost].yCoord - 128;
         float xDelta = coords[jPost].xCoord - 128;
         //float distance = sqrt((xDelta * xDelta) + (yDelta * yDelta));
         //if(distance <= weightParamPtr->getnxp()){
            //dataStart[jPost] = weightParamPtr->getStrength();
         //}
         float distance = sqrt((xDelta * xDelta) + (yDelta * yDelta));
         if(strcmp("Gaussian", weightParamPtr->getFalloffType()) == 0){
            //Gaussian fall off rate
            float variance = 1000;
            float mean = 0;
            float a = 1;
            float b = mean;
            float c = sqrt(variance);
            float x = sqrt((xDelta * xDelta) + (yDelta * yDelta));
            dataStart[jPost] = a*exp(-(pow(x - b,2) / (2 * pow(c,2)))) * weightParamPtr->getStrength();
         }
         else if(strcmp("radSquared", weightParamPtr->getFalloffType()) == 0){
            //Inverse square fall off rate
            dataStart[jPost] = (1 / (distance * distance)) * weightParamPtr->getStrength();
         }
         else if(strcmp("Log", weightParamPtr->getFalloffType()) == 0){
            float connDist = maxDistance * maxDistance; //conn->xPatchSize() * conn->xPatchSize() + conn->yPatchSize() * conn->yPatchSize();
            dataStart[jPost] = log10((1 / (distance * distance + 1)) / (1.0f / (connDist + 1))) * weightParamPtr->getStrength();
         }
         else if(strcmp("Linear", weightParamPtr->getFalloffType()) == 0){
            //Linear fall off rate (_/\_)
            dataStart[jPost] = 1 - (sqrt((xDelta * xDelta) + (yDelta * yDelta)) / maxDistance);
            dataStart[jPost] = dataStart[jPost] < 0 ? 0 : 1 - (sqrt((xDelta * xDelta) + (yDelta * yDelta)) / maxDistance);
         }
         else{
            printf("Improper falloff type specified in the params file\n");
         }
      }
   }
   return 0;
}





} /* namespace PV */
