/*
 * InitBIDSLateral.cpp
 *
 *  Created on: Aug 10, 2012
 *      Author: bnowers
 */

#include <weightinit/InitWeights.hpp>

#include <stdlib.h>
#include <math.h>
#include <iostream>

#include <include/default_params.h>
#include <io/io.h>
#include <io/fileio.hpp>
#include <utils/conversions.h>
#include <columns/InterColComm.hpp>
#include "InitBIDSLateralParams.hpp"
#include "InitBIDSLateral.hpp"


namespace PV {

InitBIDSLateral::InitBIDSLateral(char const * name, HyPerCol * hc){
   initialize_base();
   initialize(name, hc);
}

InitBIDSLateral::InitBIDSLateral(){
   initialize_base();
}

InitBIDSLateral::~InitBIDSLateral(){
}

int InitBIDSLateral::initialize_base() {
   return PV_SUCCESS;
}

int InitBIDSLateral::initialize(char const * name, HyPerCol * hc) {
   int status = InitWeights::initialize(name, hc);
   return status;
}

InitWeightsParams * InitBIDSLateral::createNewWeightParams(){
   InitWeightsParams * tempPtr = new InitBIDSLateralParams(name, parentHyPerCol);
   return tempPtr;
}

int InitBIDSLateral::calcWeights(pvdata_t * dataStart, int dataPatchIndex, int arborId){
    InitBIDSLateralParams *weightParamPtr = dynamic_cast<InitBIDSLateralParams*> (weightParams);

    if(weightParamPtr==NULL) {
       fprintf(stderr, "Failed to recast pointer to weightsParam!  Exiting...");
       exit(PV_FAILURE);
    }


    weightParamPtr->calcOtherParams(dataPatchIndex);

    //calculate the weights:
    BIDSLateralCalcWeights(dataPatchIndex, dataStart, weightParamPtr);


    return PV_SUCCESS;
}

/**
 * calculate gaussian weights between oriented line segments
 * WARNING - assumes weight and GSyn patches from task same size
 *           assumes jitterX = jitterY and patchSizex = patchSizey
*/
int InitBIDSLateral::BIDSLateralCalcWeights(int kPre, pvdata_t * dataStart, InitBIDSLateralParams * weightParamPtr) {

   //i wonder if it is possible to use ->getNXScale()??
   int nxBids = weightParamPtr->getPre()->getLayerLoc()->nx;
   int nyBids = weightParamPtr->getPre()->getLayerLoc()->ny;
   int nfBids = weightParamPtr->getPre()->getLayerLoc()->nf;
   PVHalo const * haloBids = &weightParamPtr->getPre()->getLayerLoc()->halo;

   int kPreRes = kIndexRestricted(kPre, nxBids, nyBids, nfBids, haloBids->lt, haloBids->rt, haloBids->dn, haloBids->up);
   if(kPreRes == -1){
      return 0;
   }

   //load necessary params:
   BIDSCoords * coords = weightParamPtr->getCoords();

   int arborID = 0;
   assert(arborID >= 0);
   PVPatch * weights = callingConn->getWeights(kPre, arborID);

#ifdef DEBUG_OUTPUT
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   //printf("[%d]: HyPerLayr::recvSyn: neighbor=%d num=%d actv=%p this=%p conn=%p\n", rank, neighbor, numExtended, activity, this, conn);
   printf("[%d]: HyPerLayr::recvSyn: neighbor=%d num=%d actv=%p this=%p conn=%p\n", rank, 0, numExtended, activity, this, conn);
   fflush(stdout);
#endif


   int nxGlobal = callingConn->getParent()->getNxGlobal();
   int nyGlobal = callingConn->getParent()->getNyGlobal();
   int patchSizex = nxGlobal / nxBids;
   int patchSizey = nyGlobal / nyBids;

   //BIDS space
   int nxp  = callingConn->fPatchSize() * weights->nx;
   int nyp  = callingConn->fPatchSize() * weights->ny;

   //the principle node's physical position (HyPerCol space)
   int preCoordy = coords[kPreRes].yCoord;
   int preCoordx = coords[kPreRes].xCoord;

   //the principle node's mathematical position (BIDS space)
   int kPreResy = kyPos(kPreRes, nxBids, nyBids, nfBids);
   int kPreResx = kxPos(kPreRes, nxBids, nyBids, nfBids);

   //the principle node's mathematical position (HyPerCol space)
   int patchCenterx = (kPreResx * patchSizex) + (patchSizex / 2);
   int patchCentery = (kPreResy * patchSizey) + (patchSizey / 2);

   const char * jitterSourceName = weightParamPtr->getJitterSource();
   int jitter = weightParamPtr->getJitter();
   //Divide by patch size to convert to BIDS units
   int principleJittDiffx = (jitter - abs(preCoordx - patchCenterx)) / patchSizex;
   int principleJittDiffy = (jitter - abs(preCoordy - patchCentery)) / patchSizey;

   int sy  = callingConn->getPostNonextStrides()->sy;       // stride in layer
   int sx  = callingConn->getPostNonextStrides()->sx;       // generally the # of features
   int syw = callingConn->yPatchStride(); //weights->sy;    // stride in patch

   int radius = weightParamPtr->getLateralRadius();

   //Search patches in jitter
   for (int delY = principleJittDiffy; delY < nyp - principleJittDiffy; delY++) {
      float * RESTRICT w = dataStart + delY * syw; //stride gets correct weight vector
      for (int delX = principleJittDiffx; delX < nxp - principleJittDiffx; delX++) {
         int index = callingConn->getGSynPatchStart(kPre, arborID) + delY * sy + delX * sx + 0 * 1;
         int postcoordx = coords[index].xCoord;
         int postcoordy = coords[index].yCoord;

         //Test for ranges
         int range = 2 * (radius + 2 * jitter) + 1;

         if(abs(preCoordx - postcoordx) > range || abs(preCoordy - postcoordy) > range){
            std::cout << "ERROR!  ";
            std::cout << "pre " << kPreRes << ": (" << preCoordx << ", " << preCoordy << ")  ";
            std::cout << "post " << index << ": (" << postcoordx << "," << postcoordy << ")  ";
            std::cout << "range: " << range << "\n";
            exit(-1);
         }


         float deltaX = (float)abs(postcoordx - preCoordx);
         float deltaY = (float)abs(postcoordy - preCoordy);
         float distance = sqrt((deltaX * deltaX) + (deltaY * deltaY));
         if (distance == 0){
            continue;
         }

         //the falloff rate of the conncection is determined by the params file and is chosen here
         if(strcmp("Gaussian", weightParamPtr->getFalloffType()) == 0){
            //Gaussian fall off rate
            float variance = 1000;
            float mean = 0;
            float a = 1;
            float b = mean;
            float c = sqrt(variance);
            float x = distance;
            w[delX] = a*exp(-(pow(x - b,2) / (2 * pow(c,2)))) * weightParamPtr->getStrength();
         }
         else if(strcmp("radSquared", weightParamPtr->getFalloffType()) == 0){
            //Inverse square fall off rate
            w[delX] = (1 / (distance * distance + 1)) * weightParamPtr->getStrength();
         }
         else if(strcmp("Log", weightParamPtr->getFalloffType()) == 0){
            float connDist2 = 2*radius*radius;
            float w_tmp = ((1 / (distance * distance + 1)) / (1 / (connDist2 + 1)));
            if (w_tmp > 1.0f){
               w[delX] = log10(w_tmp) * weightParamPtr->getStrength();
            }
            else{
               w[delX] = 0;
            }
         }
         else{
            printf("Improper falloff type specified in the params file\n");
         }
      }
   }

   return 0;
}


} /* namespace PV */
