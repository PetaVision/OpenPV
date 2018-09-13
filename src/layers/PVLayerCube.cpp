/*
 * PVLayerCube.cpp
 *
 *  Created on: Nov 18, 2008
 *      Author: Craig Rasmussen
 */

#include "PVLayerCube.hpp"

#include "include/default_params.h"
#include "io/io.hpp"
#include "utils/cl_random.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace PV {

PVLayerCube *pvcube_init(PVLayerCube *cube, const PVLayerLoc *loc, int numItems) {
   cube->numItems = numItems;
   cube->loc      = *loc;
   pvcube_setAddr(cube);
   return cube;
}

PVLayerCube *pvcube_new(const PVLayerLoc *loc, int numItems) {
   PVLayerCube *cube = (PVLayerCube *)calloc(pvcube_size(numItems), sizeof(char));
   assert(cube != NULL);
   pvcube_init(cube, loc, numItems);
   return cube;
}

size_t pvcube_size(int numItems) { return sizeof(PVLayerCube) + numItems * sizeof(float); }

int pvcube_delete(PVLayerCube *cube) {
   free(cube);
   return 0;
}

int pvcube_setAddr(PVLayerCube *cube) {
   cube->data = (float *)((char *)cube + sizeof(PVLayerCube));
   return 0;
}

} // namespace PV
