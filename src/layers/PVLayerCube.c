/*
 * PVLayer.c
 *
 *  Created on: Nov 18, 2008
 *      Author: Craig Rasmussen
 */

#include "PVLayerCube.h"

#include "../io/io.h"
#include "../include/default_params.h"
#include "../utils/cl_random.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef OBSOLETE // Marked Obsolete Sept 11, 2014.  Anything that calls these functions are themselves obsolete.
///////////////////////////////////////////////////////
// pvpatch interface implementation
//

// PVPatch * pvpatch_new(int nx, int ny, int nf)
PVPatch * pvpatch_new(int nx, int ny)
{
   // int sf = 1;
   // int sx = nf;
   // int sy = sx * nx;

   PVPatch * p = (PVPatch *) malloc(sizeof(PVPatch));
   assert(p != NULL);

   // pvdata_t * data = NULL;

   pvpatch_init(p, nx, ny); // pvpatch_init(p, nx, ny, nf, sx, sy, sf, data);

   return p;
}

int pvpatch_delete(PVPatch* p)
{
   free(p);
   return 0;
}

int pvpatch_inplace_delete(PVPatch* p)
{
   free(p);
   return 0;
}



#endif // OBSOLETE

///////////////////////////////////////////////////////
// PVLayerCube interface implementation
//

PVLayerCube * pvcube_init(PVLayerCube * cube, const PVLayerLoc * loc, int numItems)
{
   cube->size = pvcube_size(numItems);
   cube->numItems = numItems;
   cube->loc = *loc;
   pvcube_setAddr(cube);
   return cube;
}

PVLayerCube * pvcube_new(const PVLayerLoc * loc, int numItems)
{
   PVLayerCube * cube = (PVLayerCube*) calloc(pvcube_size(numItems), sizeof(char));
   assert(cube !=NULL);
   pvcube_init(cube, loc, numItems);
   return cube;
}

size_t pvcube_size(int numItems)
{
   size_t size = LAYER_CUBE_HEADER_SIZE;
   assert(size == EXPECTED_CUBE_HEADER_SIZE); // depends on PV_ARCH_64 setting
   return size + numItems*sizeof(float);
}

int pvcube_delete(PVLayerCube * cube)
{
   free(cube);
   return 0;
}

int pvcube_setAddr(PVLayerCube * cube)
{
   cube->data = (pvdata_t *) ((char*) cube + LAYER_CUBE_HEADER_SIZE);
   return 0;
}

#ifdef __cplusplus
}
#endif
