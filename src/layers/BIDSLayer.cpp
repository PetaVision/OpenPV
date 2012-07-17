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

int * vector;
int flag = 0;
BIDSCoords * coords; //structure array pointer that holds the randomly generated corrdinates for the specified number of BIDS nodes

namespace PV {
BIDSLayer::BIDSLayer() {
  // initialize(arguments) should *not* be called by the protected constructor.
}

BIDSLayer::BIDSLayer(const char * name, HyPerCol * hc) {
   initialize(name, hc, TypeBIDS, MAX_CHANNELS, "BIDS_update_state");
}

int BIDSLayer::initialize(const char * name, HyPerCol * hc, PVLayerType type, int num_channels, const char * kernel_name){
   LIF::initialize(name, hc, TypeBIDS, MAX_CHANNELS, "BIDS_update_state");
   float nxScale = (float)(parent->parameters()->value("BIDS_node", "nxScale"));
   float nyScale = (float)(parent->parameters()->value("BIDS_node", "nyScale"));
   int nxGlobal = (int)(parent->parameters()->value("column", "nx"));
   int nyGlobal = (int)(parent->parameters()->value("column", "ny"));
   printf("%d %d", nxGlobal, nyGlobal);
   int numNodes = (nxScale * nxGlobal) * (nyScale * nyGlobal);
   coords = (BIDSCoords*)malloc((sizeof(BIDSCoords)) * numNodes);
   getCoords(numNodes, coords);
   //vector = randomIndices(clayer->loc.nxGlobal, clayer->loc.nyGlobal);
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
//TODO create flag system for a fully populated matrix
void BIDSLayer::getCoords(int numNodes, BIDSCoords * coords){
   srand(time(NULL));
   int nxGlobal = (int)(parent->parameters()->value("column", "nx"));
   int nyGlobal = (int)(parent->parameters()->value("column", "ny"));
   int xCheck[numNodes]; //checks for repeating x indices
   int yCheck[numNodes]; //checks for repeating y indices

   for(int i = 0; i < numNodes; i++){ //zeros out the xCheck array
      xCheck[i] = 0;
      yCheck[i] = 0;
   }

   for(int i = 0; i < numNodes; i++){ //cycles through the number of nodes specified by params file
      int xIndex = 0;
      int yIndex = 0;
      int xAdded = 0;
      int yAdded = 0;
      while(xAdded == 0){ //creates a random x coordinate until one is created that has not been used yet
         xIndex = rand() % nxGlobal;
         if(xCheck[xIndex] < nxGlobal){
            //printf("xIndex: %d\t", xIndex);
            xCheck[xIndex]++;
            xAdded = 1;
         }
      }
      coords[i].xCoord = xIndex; //stores the x coordinate into the current BIDSCoord structure

      while(yAdded == 0){ //creates a random y coordinate until one is created that has not been used yet
         yIndex = rand() % nyGlobal;
         if(yCheck[yIndex] < nyGlobal){
            //printf("yIndex: %d\t", yIndex);
            yCheck[yIndex]++;
            yAdded = 1;
         }
      }
      coords[i].yCoord = yIndex; //stores the y coordinate into the current BIDSCoord structure
      //printf("(%d\n", i+1);
   }
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
