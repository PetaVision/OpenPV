/*
 * fileread.cpp
 *
 *  Created on: Aug 4, 2008
 *      Author: dcoates
 */

#include "PVLayer.h"
#include "fileread.h"
#include "../io/io.h"
#include "../include/pv_common.h"

#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <stdio.h>

typedef fileread_params params_t;

#define FILEREAD_SENTINEL -0xDC

#ifdef DEPRECATED

int fileread_init(PVLayer* l)
{
   int n, err = 0;
   params_t* params = (params_t*) l->params;
   float marginWidth = params->marginWidth;

#ifdef DEBUG_OUTPUT
   printf("[%d]: fileread_init: (kx0,ky0,nx,ny,nfeatures) = (%f, %f, %f, %f, %d)\n",
          l->columnId, l->loc.kx0, l->loc.ky0, l->loc.nx, l->loc.ny, l->numFeatures);
   fflush(stdout);
#endif

   if (params->filename != NULL) {
      err = scatterReadFile(params->filename, l, l->V, MPI_COMM_WORLD);
      if (err != 0) return err;
   }
   else {
      return -1;
   }

   // check margins

   // TODO - make sure the origin information is working correctly
   if (marginWidth != 0.0f) {
      for (n = 0; n < l->numNeurons; n++) {
         float x = xPos(n, l->xOrigin, l->dx, l->loc.nx, l->loc.ny, l->numFeatures);
         float y = yPos(n, l->yOrigin, l->dy, l->loc.nx, l->loc.ny, l->numFeatures);

         if ( x < marginWidth || x > l->loc.nxGlobal * l->dx - marginWidth ||
              y < marginWidth || y > l->loc.nyGlobal * l->dy - marginWidth ) {
            l->V[n] = 0.0;
         }
      }
   }

   if (params->invert) {
      for (n = 0; n < l->numNeurons; n++) {
         l->V[n] = (l->V[n] == 0.0) ? 1.0 : 0.0;
      }
   }

   if (params->uncolor) {
      for (n = 0; n < l->numNeurons; n++) {
         l->V[n] = (l->V[n] == 0.0) ? 0.0 : 1.0;
      }
   }

   // TODO - add retina boundary conditions

   if (l->layerId == TypeRetina - 1) {
      // 	  // i_theta = 5
      // 	  l->V[((NY/2)-2)*NX+(NX/2)+1] = 1.0;
      // 	  l->V[((NY/2)-1)*NX+(NX/2)+1] = 1.0;
      // 	  l->V[((NY/2)-1)*NX+(NX/2)-0] = 1.0;
      // 	  l->V[((NY/2)-0)*NX+(NX/2)+0] = 1.0;
      // 	  l->V[((NY/2)+1)*NX+(NX/2)+0] = 1.0;
      // 	  l->V[((NY/2)+1)*NX+(NX/2)-1] = 1.0;
      // 	  l->V[((NY/2)+2)*NX+(NX/2)-1] = 1.0;

      // i_theta = 2
      // 	  l->V[((NY/2)-2)*NX+(NX/2)-2] = 1.0;
      // 	  l->V[((NY/2)-1)*NX+(NX/2)-1] = 1.0;
      // 	  l->V[((NY/2)-0)*NX+(NX/2)+0] = 1.0;
      // 	  l->V[((NY/2)+1)*NX+(NX/2)+1] = 1.0;
      // 	  l->V[((NY/2)+2)*NX+(NX/2)+2] = 1.0;

//      l->V[((NY / 2) - 2) * NX + (NX / 2) - 2] = 1.0;
//      l->V[((NY / 2) - 1) * NX + (NX / 2) - 1] = 1.0;
//      l->V[((NY / 2) - 0) * NX + (NX / 2) + 0] = 1.0;
//      l->V[((NY / 2) + 1) * NX + (NX / 2) + 1] = 1.0;
//      l->V[((NY / 2) + 2) * NX + (NX / 2) + 2] = 1.0;
   }

   return 0;
}

void fileread_update(PVLayer *l)
{
   int k;
   static int timeStep = 0;     // static? this seems dangerous with multiple layers

   params_t* params = (params_t*) l->params;

   int stimStatus = (timeStep >= params->beginStim) && (timeStep < params->endStim);

   if (params->spikingFlag == 0.0) {
      float lProb[2];

      // copy from the V buffer to the activity buffer
      lProb[0] = stimStatus ? params->poissonEdgeProb : params->poissonBlankProb;
      lProb[1] = params->poissonBlankProb;
      for (k = 0; k < l->numNeurons; k++) {
         l->activity->data[k] = lProb[(l->V[k] > 0.0 ? 0 : 1)];
      }
   }
   else {
   // For "Poisson" spiking...
      //
      long lProb[2];

      // TODO: make this plausible--I just fudged something. (dc 8/1/08)
      //#define EDGE_PROB 1.0
      //#define BLANK_PROB 0.0001
      //lProb[0] = (double) RAND_MAX * BLANK_PROB * exp(-BLANK_PROB);
      //lProb[1] = (double) RAND_MAX * EDGE_PROB * exp(-EDGE_PROB);
      lProb[0] = stimStatus ? RAND_MAX * params->poissonEdgeProb : RAND_MAX * params->poissonBlankProb;
      lProb[1] = RAND_MAX * params->poissonBlankProb;

      for (k = 0; k < l->numNeurons; k++) {
         // If the image edge (stored in V) is set, rand is compared against lProb[0]
         // else it's compared against lProb[1]. If the rand val below that, fire,
         // with amplitude 1.0.
         l->activity->data[k]
                 = (rand() > lProb[(l->V[k] > 0.0 ? 0 : 1)]) ? 0.0 : 1.0;
      }
   }
   if (l->yOrigin == 0) timeStep++;

}

#endif
