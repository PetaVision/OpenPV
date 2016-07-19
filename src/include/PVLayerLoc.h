/*
 * PVLayerLoc.h
 */

#ifndef PVLAYERLOC_H_
#define PVLAYERLOC_H_

/**
 * PVHalo describes the padding for a layer.  Padding must
 * be at least the number of boundary cells, nb, but may be
 * more if needed to make memory operations more efficient,
 * for example, to align memory to a vector size.
 */
typedef struct PVHalo_ {
   int lt, rt, dn, up;  // padding in {left, right, down, up} directions
} PVHalo;

/**
 * PVLayerLoc describes the local location of a layer within the global space
 */
typedef struct PVLayerLoc_ {
   int nbatch, nx, ny, nf; // local number of grid pts in each dimension
   int nbatchGlobal, nxGlobal, nyGlobal; // total number of grid pts in the global space
   int kb0, kx0, ky0;      // origin of the local layer in global index space
   PVHalo halo;            // padding for memory (must include nb)
} PVLayerLoc;

#endif /* PVLAYERLOC_H_ */
