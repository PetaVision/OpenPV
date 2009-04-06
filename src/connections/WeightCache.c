/*
 * weight_cache.c
 *
 *  Created on: Aug 27, 2008
 *      Author: dcoates
 */

#include "../include/pv_common.h"
#include "PVConnection.h"
#include "WeightCache.h"
#include <stdlib.h>
#include <math.h> // for fabs
int PV_weightCache_getNumPreKernels(PVConnection *con)
{
   float xRatio = (con->post->loc.dx / con->pre->loc.dx);
   float yRatio = (con->post->loc.dy / con->pre->loc.dy);
   if (xRatio < 1) xRatio = 1;
   if (yRatio < 1) yRatio = 1;
   return xRatio * yRatio * con->pre->numFeatures;
}

int PV_weightCache_getNumPostKernels(PVConnection *con)
{
   float xRatio = (con->pre->loc.dx / con->post->loc.dx);
   float yRatio = (con->pre->loc.dy / con->post->loc.dy);
   if (xRatio < 1) xRatio = 1;
   if (yRatio < 1) yRatio = 1;
   return xRatio * yRatio * con->post->numFeatures;
}

int PV_weightCache_init(PVConnection * con)
{
   int n, m;
   int numCache = con->post->numFeatures * con->post->loc.nx * con->post->loc.ny;

   con->numKernels = PV_weightCache_getNumPreKernels(con);

   con->xStride = con->post->numFeatures;
   con->yStride = con->post->numFeatures * con->post->loc.nx;

   con->numPostSynapses = (int*) calloc(con->numKernels, sizeof(int));
   con->weights = (float**) malloc(sizeof(float*) * con->numKernels);
   con->postCacheIndices = (int**) malloc(sizeof(int*) * con->numKernels);
   con->postKernels = (int**) malloc(sizeof(int*) * con->numKernels);
   // worst case: assume fully connected. TODO: dynamically allocate
   for (n = 0; n < con->numKernels; n++) {
      con->weights[n] = (float*) malloc(sizeof(int) * numCache);
      con->postCacheIndices[n] = (int*) malloc(sizeof(int) * numCache);
      con->postKernels[n] = (int*) malloc(sizeof(int) * numCache);
   }

   // init to sentinel
   for (m = 0; m < con->numKernels; m++)
      for (n = 0; n < numCache; n++)
         con->weights[m][n] = SENTINEL;

   return 0;
}

int PV_weightCache_finalize(PVConnection * con)
{
   int n;

   for (n = 0; n < con->numKernels; n++) {
      free(con->weights[n]);
      free(con->postCacheIndices[n]);
      free(con->postKernels[n]);
   }
   free(con->weights);
   free(con->postCacheIndices);
   free(con->postKernels);
   free(con->numPostSynapses);

   con->weights          = NULL;
   con->postCacheIndices = NULL;
   con->postKernels      = NULL;
   con->numPostSynapses  = NULL;

   return 0;
}
