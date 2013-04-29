/*
 * InitBIDSLateral.cpp
 *
 *  Created on: Aug 10, 2012
 *      Author: bnowers
 */

#include "InitWeights.hpp"

#include <stdlib.h>
#include <math.h>

#include "../include/default_params.h"
#ifdef OBSOLETE // Marked obsolete Feb. 27, 2012.  Replaced by PatchProbe.
#include "../io/ConnectionProbe.hpp"
#endif // OBSOLETE
#include "../io/io.h"
#include "../io/fileio.hpp"
#include "../utils/conversions.h"
#include "../utils/pv_random.h"
#include "../columns/InterColComm.hpp"
#include "InitBIDSLateralParams.hpp"
#include "InitBIDSLateral.hpp"


namespace PV {

InitBIDSLateral::InitBIDSLateral(){
   initialize_base();
}

InitBIDSLateral::~InitBIDSLateral(){
   // TODO Auto-generated destructor stub
}

/*This method does the three steps involved in initializing weights.  Subclasses shouldn't touch this method.
 * Subclasses should only generate their own versions of calcWeights to do their own type of weight initialization.
 *
 * This method initializes the full unshrunken patch.  The input argument numPatches is ignored.  Instead, method uses getNumDataPatches to determine number of
 * data patches.
 */
int InitBIDSLateral::initializeWeights(PVPatch *** patches, pvdata_t ** dataStart, int numPatches, const char * filename, HyPerConn * callingConn, double * timef /*default NULL*/){
   PVParams * inputParams = callingConn->getParent()->parameters();
   movieLayer = (BIDSMovieCloneMap*)(callingConn->getParent()->getLayerFromName("BIDS_Movie"));
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
//      std::cout<<"\n\nnumAbors: " << numArbors << " Patch size: " << callingConn->getNumDataPatches() << "\n";
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

InitWeightsParams * InitBIDSLateral::createNewWeightParams(HyPerConn * callingConn){
   InitWeightsParams * tempPtr = new InitBIDSLateralParams(callingConn);
   return tempPtr;
}

int InitBIDSLateral::calcWeightsBIDS(/* PVPatch * patch */ pvdata_t * dataStart, int dataPatchIndex, int arborId, InitWeightsParams *weightParams, HyPerConn * conn){
    InitBIDSLateralParams *weightParamPtr = dynamic_cast<InitBIDSLateralParams*> (weightParams);

    if(weightParamPtr==NULL) {
       fprintf(stderr, "Failed to recast pointer to weightsParam!  Exiting...");
       exit(PV_FAILURE);
    }


    weightParamPtr->calcOtherParams(dataPatchIndex);

    //calculate the weights:
    BIDSLateralCalcWeights(dataPatchIndex, dataStart, weightParamPtr, conn);


    return PV_SUCCESS;
}

int InitBIDSLateral::initialize_base() {
   return PV_SUCCESS;
}



/**
 * calculate gaussian weights between oriented line segments
 * WARNING - assumes weight and GSyn patches from task same size
 *           assumes jitterX = jitterY and patchSizex = patchSizey
*/
int InitBIDSLateral::BIDSLateralCalcWeights(/* PVPatch * patch */ int kPre, pvdata_t * dataStart, InitBIDSLateralParams * weightParamPtr, HyPerConn * conn) {

   //i wonder if it is possible to use ->getNXScale()??
   int nxBids = weightParamPtr->getPre()->getLayerLoc()->nx;
   int nyBids = weightParamPtr->getPre()->getLayerLoc()->ny;
   int nfBids = weightParamPtr->getPre()->getLayerLoc()->nf;
   int nbBids = weightParamPtr->getPre()->getLayerLoc()->nb;

   //std::cout << kPre << "\n";
   int kPreRes = kIndexRestricted(kPre, nxBids, nyBids, nfBids, nbBids);
//   std::cout << kPreRes << "\n";
   if(kPreRes == -1){
      return 0;
   }

   //load necessary params:
   BIDSCoords * coords = weightParamPtr->getCoords();

   int arborID = 0;
   assert(arborID >= 0);
   PVPatch * weights = conn->getWeights(kPre, arborID);

#ifdef DEBUG_OUTPUT
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   //printf("[%d]: HyPerLayr::recvSyn: neighbor=%d num=%d actv=%p this=%p conn=%p\n", rank, neighbor, numExtended, activity, this, conn);
   printf("[%d]: HyPerLayr::recvSyn: neighbor=%d num=%d actv=%p this=%p conn=%p\n", rank, 0, numExtended, activity, this, conn);
   fflush(stdout);
#endif


//      std::cout <<"InitBIDSLateral: nxy, nyp: " << nxp << ", " << nyp << "\n";
      int nxGlobal = conn->getParent()->getNxGlobal();
      int nyGlobal = conn->getParent()->getNyGlobal();
      int patchSizex = nxGlobal / nxBids;
      int patchSizey = nyGlobal / nyBids;

      //BIDS space
      int nxp  = conn->fPatchSize() * weights->nx;
      int nyp  = conn->fPatchSize() * weights->ny;

//      std::cout << "InitBIDSLat: (" << nxp << ", " << nyp << ")\n";

      //the principle node's physical position (HyPerCol space)
      int preCoordy = coords[kPreRes].yCoord;
      int preCoordx = coords[kPreRes].xCoord;

//      std::cout << "InitBIDSLateral: Index " << kPreRes << ": (" << preCoordx << ", " << preCoordy << ")\n";

      //the principle node's mathematical position (BIDS space)
      int kPreResy = kyPos(kPreRes, nxBids, nyBids, nfBids);
      int kPreResx = kxPos(kPreRes, nxBids, nyBids, nfBids);

      //the principle node's mathematical position (HyPerCol space)
      int patchCenterx = (kPreResx * patchSizex) + (patchSizex / 2);
      int patchCentery = (kPreResy * patchSizey) + (patchSizey / 2);

      const char * jitterSourceName = conn->getParent()->parameters()->stringValue(conn->getName(), "jitterSource");
      int jitter = (int)weightParamPtr->getPre()->getParent()->parameters()->value(jitterSourceName, "jitter");
      //Divide by patch size to convert to BIDS units
      int principleJittDiffx = (jitter - abs(preCoordx - patchCenterx)) / patchSizex;
      int principleJittDiffy = (jitter - abs(preCoordy - patchCentery)) / patchSizey;
/*
      std::cout << "principleJittDiffx = " << principleJittDiffx << " :(jitter - abs(preCoordx - patchCenterx))/patchSizex : (" << jitter <<" - abs(" << preCoordx << " - " << patchCenterx << ")) / " << patchSizex << "\n";
      std::cout << "principleJittDiffy = " << principleJittDiffy << " :(jitter - abs(preCoordy - patchCentery))/patchSizey : (" << jitter <<" - abs(" << preCoordy << " - " << patchCentery << ")) / " << patchSizey << "\n";
*/
      int sy  = conn->getPostNonextStrides()->sy;       // stride in layer
      int sx  = conn->getPostNonextStrides()->sx;       // generally the # of features
      int syw = conn->yPatchStride(); //weights->sy;    // stride in patch
      pvdata_t * channel = conn->getPost()->getChannel(conn->getChannel());

//      std::cout << "Loop: " << principleJittDiffy << " to " << nyp - principleJittDiffy << "\n";
//      std::cout << "Loop: " << principleJittDiffx << " to " << nxp - principleJittDiffx << "\n";
//      std::cout << "pre: (" << preCoordx << ", " << preCoordy << ")\n";
      //Search patches in jitter
//      int maxindex = 0;
      for (int delY = principleJittDiffy; delY < nyp - principleJittDiffy; delY++) {
         float * RESTRICT w = dataStart + delY * syw; //stride gets correct weight vector
         for (int delX = principleJittDiffx; delX < nxp - principleJittDiffx; delX++) {
//            pvdata_t * memLoc = conn->getGSynPatchStart(kPre, arborID) + delY * sy + delX * sx + 0 * 1;
//            int index = memLoc - channel;
            int index = conn->getGSynPatchStart(kPre, arborID) + delY * sy + delX * sx + 0 * 1;
            /*
            if (index >= maxindex){
               maxindex = index;
            }
            */
            int postcoordx = coords[index].xCoord;
            int postcoordy = coords[index].yCoord;
 //           std::cout << "\tpost: (" << postcoordx << "," << postcoordy << ")\n";



            //Test for ranges
            int radius = (int)conn->getParent()->parameters()->value(conn->getName(), "lateralRadius");
            int range = 2 * (radius + 2 * jitter) + 1;

            if(abs(preCoordx - postcoordx) > range || abs(preCoordy - postcoordy) > range){
               std::cout << "ERROR!  ";
               std::cout << "pre " << kPreRes << ": (" << preCoordx << ", " << preCoordy << ")  ";
               std::cout << "post " << index << ": (" << postcoordx << "," << postcoordy << ")  ";
               std::cout << "range: " << range << "\n";
               exit(-1);
            }


            float deltaX = fabs(postcoordx - preCoordx);
            float deltaY = fabs(postcoordy - preCoordy);
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
               int lateralRad = (int)conn->getParent()->parameters()->value(conn->getName(), "lateralRadius");
               float connDist2 = 2*lateralRad*lateralRad;
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
