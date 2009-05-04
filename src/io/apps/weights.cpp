/*
 * weights.cpp
 *
 *  Created on: May 1, 2009
 *      Author: rasmussn
 */

#include "../../include/pv_types.h"
#include "../io.h"
#include <assert.h>

#define NUM_PARAMS 6

int main(int argc, char * argv[])
{
   int err = 0;
   int numParams, nx, ny, nf;
   int params[NUM_PARAMS];

   assert(argc == 2);
   FILE * fp = pv_open_binary(argv[1], &numParams, &nx, &ny, &nf);
   assert(fp != NULL);
   assert(numParams == NUM_PARAMS);

   assert(pv_read_binary_params(fp, numParams, params) == numParams);

   float minVal = (float) params[3];
   float maxVal = (float) params[4];
   int numPatches = params[5];

   int numItems = nx * ny * nf;  // maximum number of items in a patch
   int patchSize = sizeof(PVPatch) + numItems * sizeof(pvdata_t);

   PVPatch ** patches = (PVPatch **) malloc(numPatches * sizeof(PVPatch *));
   unsigned char * patchBuf = (unsigned char *) malloc(numPatches * patchSize);

   for (int i = 0; i < numPatches; i++) {
      patches[i] = (PVPatch *) &patchBuf[i*patchSize];
      PVPatch * p = patches[i];
      p->data = (pvdata_t *) ((unsigned char *) p + sizeof(PVPatch));
   }

   err = pv_read_patches(fp, nf, minVal, maxVal, patches, numPatches);

   PVPatch * p = patches[0];
   for (int i = 0; i < p->nx; i++) {
      printf("w[%d]=%f\n", i, p->data[i]);
   }

   pv_close_binary(fp);

   free(patchBuf);
   free(patches);

   return err;
}
