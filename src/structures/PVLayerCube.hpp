#ifndef PVLAYERCUBE_HPP_
#define PVLAYERCUBE_HPP_

#include "include/PVLayerLoc.h"

/**
 * PVLayerCube is a 3D cube (features,x,y) of a layer's data,
 *    plus location information
 */
struct PVLayerCube {
   int numItems; // number of items in data buffer
   float const *data; // pointer to data
   PVLayerLoc loc;
   bool isSparse;
   long const *numActive;
   void const *activeIndices;
};

#endif // PVLAYERCUBE_HPP_
