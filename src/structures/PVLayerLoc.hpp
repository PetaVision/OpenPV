/*
 * PVLayerLoc.hpp
 */

#ifndef PVLAYERLOC_HPP_
#define PVLAYERLOC_HPP_

/**
 * PVHalo describes the padding for a layer.  Padding must
 * be at least the number of boundary cells, nb, but may be
 * more if needed to make memory operations more efficient,
 * for example, to align memory to a vector size.
 */
struct PVHalo {
   int lt, rt, dn, up; // padding in {left, right, down, up} directions
};

/**
 * PVLayerLoc describes the local location of a layer within the global space
 */
struct PVLayerLoc {
   int nbatch, nx, ny, nf; // local number of grid pts in each dimension
   int nbatchGlobal, nxGlobal, nyGlobal; // total number of grid pts in the global space
   int kb0, kx0, ky0; // origin of the local layer in global index space
   PVHalo halo; // padding for memory (must include nb)
};

#endif /* PVLAYERLOC_HPP_ */
