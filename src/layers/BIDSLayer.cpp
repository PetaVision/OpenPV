/*
 * BIDSLayer.cpp
 *
 *  Created on: Jun 26, 2012
 *      Author: Bren Nowers
 */

#include "HyPerLayer.hpp"
#include "BIDSLayer.hpp"
#include "LIF.hpp"
#include "../kernels/BIDS_update_state.cl"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

int * vector;
int flag = 0;

namespace PV {
BIDSLayer::BIDSLayer() {
  // initialize(arguments) should *not* be called by the protected constructor.
}

BIDSLayer::BIDSLayer(const char * name, HyPerCol * hc) {
   initialize(name, hc, TypeBIDS, MAX_CHANNELS, "BIDS_update_state");
}

int BIDSLayer::initialize(const char * name, HyPerCol * hc, PVLayerType type, int num_channels, const char * kernel_name){
   LIF::initialize(name, hc, TypeBIDS, MAX_CHANNELS, "BIDS_update_state");
   float nxScale = (float)(parent->parameters()->value(name, "nxScale"));
   float nyScale = (float)(parent->parameters()->value(name, "nyScale"));
   int jitter = (int)(parent->parameters()->value(name, "jitter"));
   int HyPerColWidth = (int)(parent->parameters()->value("column", "nx"));
   int HyPerColHeight = (int)(parent->parameters()->value("column", "ny"));
   numNodes = (nxScale * HyPerColWidth) * (nyScale * HyPerColHeight);
   coords = (BIDSCoords*)malloc((sizeof(BIDSCoords)) * numNodes);
   setCoords(numNodes, coords, jitter);
   return PV_SUCCESS;
}

int BIDSLayer::updateState(float time, float dt)
{
   int status = 0;
   update_timer->start();

#ifdef PV_USE_OPENCL
   if((gpuAccelerateFlag)&&(true)) {
      updateStateOpenCL(time, dt);
   }
   else {
#endif
      const int nx = clayer->loc.nx;
      const int ny = clayer->loc.ny;
      const int nf = clayer->loc.nf;
      const int nb = clayer->loc.nb;

      pvdata_t * GSynHead   = GSyn[0];
//      pvdata_t * GSynExc   = getChannel(CHANNEL_EXC);
//      pvdata_t * GSynInh   = getChannel(CHANNEL_INH);
//      pvdata_t * GSynInhB  = getChannel(CHANNEL_INHB);
      pvdata_t * activity = clayer->activity->data;

      BIDS_update_state(vector, flag, getNumNeurons(), time, dt, nx, ny, nf, nb, &lParams, rand_state, clayer->V, Vth, G_E, G_I, G_IB, GSynHead, activity);
#ifdef PV_USE_OPENCL
   }
#endif

   updateActiveIndices();
   update_timer->stop();
   return status;
}

//This function fills an array with structures of random, non-repeating coordinates at which BIDS nodes will be "placed"
void BIDSLayer::setCoords(int numNodes, BIDSCoords * coords, int jitter){
   srand(time(NULL));
   assert(jitter >= 0); //jitter cannot be below zero
   int HyPerColWidth = (int)(parent->parameters()->value("column", "nx")); //the length of a side of the HyPerColumn
   int layerWidth = (int)(sqrt(numNodes)); //the length of a size of the BIDSLayer
   int patchSize = (HyPerColWidth / layerWidth); //the length of a side of a patch in the HyPerColumn
   int patchMidx = patchSize / 2;
   int patchMidy = patchSize / 2;
   int jitterRange = jitter * 2;

   int lowerboundx = patchMidx - jitter;
   int lowerboundy = patchMidy - jitter;
   for(int i = 0; i < numNodes; i++){ //cycles through the number of nodes specified by params file
      if(patchMidx <= HyPerColWidth - (patchSize / 2)) { //iterator moves to the next column in the current row
         if(jitter == 0){ //if jitter is 0, then the nodes should be placed in the origin of each patch
            coords[i].xCoord = patchMidx;
            coords[i].yCoord = patchMidy;
         }
         else{
            coords[i].xCoord = rand() % jitterRange + lowerboundx; //stores the x coordinate into the current BIDSCoord structure
            coords[i].yCoord = rand() % jitterRange + lowerboundy; //stores the y coordinate into the current BIDSCoord structure
            lowerboundx += patchSize;
         }
      }
      else{ //iterator moves to the next row in the matrix
         patchMidy += patchSize;
         patchMidx = patchSize / 2;
         if(jitter == 0){
            coords[i].xCoord = patchMidx;
            coords[i].yCoord = patchMidy;
         }
         else{
            lowerboundx = patchMidx - jitter;
            lowerboundy += patchSize;
            coords[i].xCoord = rand() % jitterRange + lowerboundx; //stores the x coordinate into the current BIDSCoord structure
            coords[i].yCoord = rand() % jitterRange + lowerboundy; //stores the y coordinate into the current BIDSCoord structure
            lowerboundx += patchSize;
         }
      }
      patchMidx += patchSize;
   }

   //for(int i = 0; i < numNodes; i++){
      //printf("%d,%d ", coords[i].xCoord, coords[i].yCoord);
   //}
}

BIDSCoords * BIDSLayer::getCoords(){
   return coords;
}

int * BIDSLayer::randomIndices(int numMatrixCol, int numMatrixRow){
   int numNodes = (int)(parent->parameters()->value("BIDS_node", "numNodes"));
   srand(time(NULL));
   int i = 0;
   int numPixels = numMatrixCol * numMatrixRow;
   int vector_tmp[numPixels];
   int index;
   int upperb = numPixels - 1;
   int lowerb = 0;

   int max = 0;

   if(flag){
      max = numNodes;
   } else {
      max = numPixels - numNodes;
   }

   for(int j = 0; j < numPixels; j++){
      if (flag) {
         vector_tmp[j] = 0;
      }
      else{
         vector_tmp[j] = 1;
      }
   }

   while(i < numNodes) {
      index = (rand() % (upperb - lowerb)) + lowerb; //creates a random index in which to place a node based on the the upper and lower bounds

      if(flag){
         if(vector_tmp[index] == 0){
            vector_tmp[index] = 1;
            i++;
         }
      }
      else{
         if(vector_tmp[index] == 1){
            vector_tmp[index] = 0;
            i++;
         }
      }
   }
   int * vecPtr = vector_tmp;
   return vecPtr;
}

void BIDSLayer::findCoordinates(int numMatrixCol, int numMatrixRow){
   int i;
   //int numNodes = (int)(parent->parameters()->value("BIDS_receptor", "numNodes"));
   int numPixels = numMatrixCol * numMatrixRow;
   int coord[numPixels][2];
   int weightMatrix[numMatrixCol][numMatrixRow];
   int col;
   int row;
   int count;

   for(i = 0; i < numPixels; i++){
      count = 1;
      if(flag){
         if(vector[i] == 1){
            col = i % numMatrixCol;
            if(col == 0){
               col = numMatrixCol;
            }
            coord[i][0] = col;
            for(int j = numMatrixCol; j < numPixels; j += numMatrixCol){
               if(i >= j){
                  count++;
               }
            }
            row = count;
            coord[i][1] = row;
         }
      }
      else{
         if(vector[i] == 0){
            col = i % numMatrixCol;
            if(col == 0){
               col = numMatrixCol;
            }
            coord[i][0] = col;
            for(int j = numMatrixCol; j < numPixels; j += numMatrixCol){
               if(i >= j){
                  count++;
               }
            }
            row = count;
            coord[i][1] = row;
         }
      }
   }

   //zeros out the weight matrix for safety purposes
   for(int i = 0; i < numMatrixCol; i++){
      for(int j = 0; j < numMatrixRow; j++){
         weightMatrix[i][j] = 0;
         weightMatrix[j][i] = 0;
      }
   }

   for(int i = 0; i < numMatrixCol; i++){
      for(int j = 0; j < numMatrixRow; j++){
         if((coord[i][0] == i) && (coord[i][1] == j)){
            weightMatrix[i][j] = findWeights(coord[i][0], coord[i][1], coord[j][0], coord[j][1]);
            weightMatrix[j][i] = findWeights(coord[i][0], coord[i][1], coord[j][0], coord[j][1]);
         }
      }
   }
}

int BIDSLayer::findWeights(int x1, int y1, int x2, int y2){
   int distance;
   //int numNodes = (int)(parent->parameters()->value("BIDS_receptor", "numNodes"));
   return distance;
}

}
