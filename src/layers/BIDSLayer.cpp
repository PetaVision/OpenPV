/*
 * BIDSLayer.cpp
 *
 *  Created on: Jun 26, 2012
 *      Author: Bren Nowers
 */

#include "HyPerLayer.hpp"
#include "BIDSLayer.hpp"
#include "LIF.hpp"
#include "../kernels/LIF_update_state.cl"

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

      //BIDS_update_state(vector, flag, getNumNeurons(), time, dt, nx, ny, nf, nb, &lParams, rand_state, clayer->V, Vth, G_E, G_I, G_IB, GSynHead, activity);
      LIF_update_state(getNumNeurons(), time, dt, nx, ny, nf, nb, &lParams, rand_state, clayer->V, Vth, G_E, G_I, G_IB, GSynHead, activity);
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
   int layerWidth = (int)(sqrt(numNodes)); //the length of a size of the BIDSLayer //TODO: fix this to not assume square bids node
   int patchSize = (HyPerColWidth / layerWidth); //the length of a side of a patch in the HyPerColumn
   int patchMidx = patchSize / 2;
   int patchMidy = patchSize / 2;
   int jitterRange = jitter * 2;

   //TODO: Set up physical position for margin nodes
   int lowerboundx = 0;
   int lowerboundy = 0;
   for(int i = 0; i < numNodes; i++){ //cycles through the number of nodes specified by params file
      if(lowerboundx <= HyPerColWidth - (patchSize / 2)) { //iterator moves to the next column in the current row
         if(jitter == 0){ //if jitter is 0, then the nodes should be placed in the middle of each patch
            coords[i].xCoord = patchMidx + (i * patchSize % HyPerColWidth);
            coords[i].yCoord = patchMidy + int(i * patchSize / HyPerColWidth) * patchSize;
         }
         else{
            int jitX = rand() % jitterRange - jitter; //stores the x coordinate into the current BIDSCoord structure
            int jitY = rand() % jitterRange - jitter; //stores the y coordinate into the current BIDSCoord structure
            coords[i].xCoord = lowerboundx + patchMidx + jitX; //stores the x coordinate into the current BIDSCoord structure
            coords[i].yCoord = lowerboundy + patchMidy + jitY; //stores the y coordinate into the current BIDSCoord structure
            lowerboundx += patchSize;
         }
      }
      else{ //iterator moves to the next row in the matrix
         if(jitter == 0){
            coords[i].xCoord = patchMidx + (i * patchSize % HyPerColWidth);
            coords[i].yCoord = patchMidy + int(i * patchSize / HyPerColWidth) * patchSize;
         }
         else{
            lowerboundx = 0;
            lowerboundy += patchSize;
            int jitX = rand() % jitterRange - jitter; //stores the x coordinate into the current BIDSCoord structure
            int jitY = rand() % jitterRange - jitter; //stores the y coordinate into the current BIDSCoord structure
            coords[i].xCoord = lowerboundx + patchMidx + jitX; //stores the x coordinate into the current BIDSCoord structure
            coords[i].yCoord = lowerboundy + patchMidy + jitY; //stores the y coordinate into the current BIDSCoord structure
            lowerboundx += patchSize;
         }
      }
   }

   //for(int i = 0; i < numNodes; i++){
      //printf("[x,y] = [%d,%d]\n", coords[i].xCoord, coords[i].yCoord);
   //}
}

BIDSCoords * BIDSLayer::getCoords(){
   return coords;
}


}
