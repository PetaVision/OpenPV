/*
 * conversions.hpp
 *
 *  Created on: Jan 7, 2010
 *      Author: rasmussn
 */

#ifndef CONVERSIONS_H_
#define CONVERSIONS_H_

#include "include/PVLayerLoc.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

#ifdef PV_RUN_ON_GPU
#define CONVERSIONS_SPECIFIER __device__
#define CONVERSIONS_NAMESPACE PVCuda
#else
#define CONVERSIONS_SPECIFIER static
#define CONVERSIONS_NAMESPACE PV
#endif // PV_RUN_ON_GPU

namespace CONVERSIONS_NAMESPACE {

/**
 * compute distance from kzPre to the nearest kzPost, i.e.
 *    (xPost - xPre) or (yPost - yPre)
 * in units of both pre- and post-synaptic dx (or dy).
 *
 * distance can be positive or negative
 * zLog2ScaleDiff in the x-direction is post->getXScale() - pre->getXScale() = log2(nxPre/nxPost);
 * analogously in y-direction.
 *
 * returns kzPost, which is local x (or y) index of nearest cell in post layer
 */
CONVERSIONS_SPECIFIER inline int
dist2NearestCell(int kzPre, int zLog2ScaleDiff, float *distPre, float *distPost) {
   if (zLog2ScaleDiff == 0) {
      // one-to-one case
      *distPre  = 0.0f;
      *distPost = 0.0f;
      return kzPre;
   }
   else if (zLog2ScaleDiff > 0) {
      // many-to-one case
      float scaleFactor       = powf(2.0f, (float)zLog2ScaleDiff);
      float kzPreToPostCoords = ((float)kzPre - 0.5f * (scaleFactor - 1.0f)) / scaleFactor;
      float kzPost            = round(kzPreToPostCoords);
      *distPost               = kzPost - kzPreToPostCoords;
      *distPre                = *distPost * scaleFactor;
      return (int)kzPost;
   }
   else {
      assert(zLog2ScaleDiff < 0);
      // one-to-many case
      float scaleFactor = powf(2.0f, (float)(-zLog2ScaleDiff));
      *distPost         = -0.5f;
      *distPre          = -0.5f / scaleFactor;
      return (int)(((float)kzPre + 0.5f) * scaleFactor) - 1;
      // left neighbor, add 1 for right neighbor
   }
}

/**
 * Return the leading index in z direction (either x or y) of a patch in postsynaptic layer
 * @kzPre is the pre-synaptic index in z direction (can be either local or global)
 * @nzPatch is the size of patch in z direction
 * @zLog2ScaleDiff is the relative scale factor log2(nzPre / nzPost).
 *
 * kzPre is always in restricted coordinates.
 */
CONVERSIONS_SPECIFIER inline int zPatchHead(int kzPre, int nzPatch, int zLog2ScaleDiff) {
   if (zLog2ScaleDiff == 0) {
      // one-to-one case
      return kzPre - (nzPatch - 1) / 2; // integer arithmetic
   }
   else if (zLog2ScaleDiff > 0) {
      // many-to-one case
      float tstride         = powf(2.0f, (float)zLog2ScaleDiff);
      float halfWidth       = 0.5f * (float)(nzPatch - 1.0f);
      float zPreInPostSpace = ((float)kzPre + 0.5f) / tstride;
      return (int)floor(zPreInPostSpace - halfWidth);
   }
   else {
      assert(zLog2ScaleDiff < 0);
      // one-to-many case
      int stride = (int)powf(2.0f, -zLog2ScaleDiff);
      return kzPre * stride - (nzPatch - stride) / 2;
      // A note regarding integer arithmetic. stride must be even here, and the typical use case
      // is that nzPatch is an integer multiple of stride; then there is no truncation from
      // integer division. If nzPatch is odd, the result is the same as if nzPatch-1 were given.
   }
}

/*
 * The following functions are simple, static inline functions.  They have been given the
 * compiler directive elemental (with same semantics as in Fortran).  The elemental functions are
 * declared in terms of scalar arguments, but can take and return arrays.  Elemental functions
 * are vectorizable.
 *
 * Notation:
 *
 *   scaleLog2 - absolute distance scale (between neurons) of a layer relative to retina
 *     - e.g. if xScaleLog2 == 1 then dx == 2, if xScaleLog2 == -1 then dx == 1/2
 *
 */

//! RETURNS FEATURE INDEX FROM LINEAR INDEX
/**
 * Return the feature index for the given k index
 * @k the k index (can be either global or local depending on if nx,ny are global or local)
 * @nx the number of neurons in the x direction
 * @ny the number of neurons in the y direction
 * @nf the number of neurons in the feature direction
 * REMARKS:
 *      - since k = ky * (nf*nx) + kx * nf + kf, we easily see that
 *      kf = mod(k,nf), i.e. kf it is the reminder of the division of k by nf,
 *      since kf <= nf-1.
 *      .
 */
CONVERSIONS_SPECIFIER inline int featureIndex(int k, int nx, int ny, int nf) { return k % nf; }

//! RETURNS X INDEX FROM LINEAR INDEX
/*!
 * Return the position kx for the given k index
 * @k the k index (can be either global or local depending on if nx,ny are global or local)
 * @nx the number of neurons in the x direction
 * @ny the number of neurons in the y direction
 * @nf the number of neurons in the feature direction
 * REMARKS:
 *   - since k = ky * (nf*nx) + kx * nf + kf, we easily see first that
 *    a = k/nf = ky*nx + kx, and then that
 *    kx = mod(a,nx), i.e. kx is the reminder of the division of a by nx,
 *    since kx <= nx-1.
 *    .
 */
CONVERSIONS_SPECIFIER inline int kxPos(int k, int nx, int ny, int nf) { return (k / nf) % nx; }

//! RETURNS Y INDEX FROM LINEAR INDEX
/*!
 * Return the position ky for the given k index
 * @k the k index (can be either global or local depending on if nx,ny are global or local)
 * @nx the number of neurons in the x direction
 * @ny the number of neurons in the y direction
 * @nf the number of neurons in the feature direction
 * REMARKS:
 *   - since k = ky * (nf*nx) + kx * nf + kf, we easily see first that
 *    kx = floor(k/(nx*nf)) since kx*nf + kf < nx*nf
 *    (note that kx <= nx-1 and kf <= nf-1).
 *   .
 */
//#pragma FTT elemental, vectorize
CONVERSIONS_SPECIFIER inline int kyPos(int k, int nx, int ny, int nf) { return k / (nx * nf) % ny; }

//! RETURNS B INDEX FROM LINEAR INDEX
/*!
 * Return the position ky for the given k index
 * @k the k index (can be either global or local depending on if nx,ny are global or local)
 * @nx the number of neurons in the x direction
 * @ny the number of neurons in the y direction
 * @nf the number of neurons in the feature direction
 */
//#pragma FTT elemental, vectorize
CONVERSIONS_SPECIFIER inline int batchIndex(int k, int nb, int nx, int ny, int nf) {
   return k / (nx * nf * ny);
}

/**
 * Returns the x dimension scale length for the layer in retinatopic units
 * where dx == 1
 * @xScaleLog2 the log2 scale factor for the layer
 *     - e.g. if xScaleLog2 == 1 then dx == 2, if xScaleLog2 == -1 then dx == 1/2
 */
CONVERSIONS_SPECIFIER inline float deltaX(int xScaleLog2) { return powf(2.0f, (float)xScaleLog2); }

/**
 * Returns the y dimension scale length for the layer in retinatopic units
 * where dy == 1
 * @yScaleLog2 the log2 scale factor for the layer
 *     - e.g. if yScaleLog2 == 1 then dy == 2, if yScaleLog2 == -1 then dy == 1/2
 */
CONVERSIONS_SPECIFIER inline float deltaY(int yScaleLog2) { return powf(2.0f, (float)yScaleLog2); }

/**
 * Returns the _global_ x origin in retinatopic units where dx == 1
 * @xScaleLog2 the log2 scale factor for the layer
 *     - e.g. if xScaleLog2 == 1 then dx == 2, if xScaleLog2 == -1 then dx == 1/2
 */
CONVERSIONS_SPECIFIER inline float xOriginGlobal(int xScaleLog2) {
   return 0.5f * deltaX(xScaleLog2);
}

/**
 * Returns the _global_ y origin in retinatopic units where dy == 1
 * @yScaleLog2 the log2 scale factor for the layer
 *     - e.g. if yScaleLog2 == 1 then dy == 2, if yScaleLog2 == -1 then dy == 1/2
 */
CONVERSIONS_SPECIFIER inline float yOriginGlobal(int yScaleLog2) {
   return 0.5f * deltaY(yScaleLog2);
}

/**
 * Returns the global x position in physical space
 * @kGlobal the global k index
 * @xScaleLog2 the log2 scale factor for the layer
 *     - e.g. if xScaleLog2 == 1 then dx == 2, if xScaleLog2 == -1 then dx == 1/2
 * @nxGlobal the number of global lattice sites in the x direction
 * @nyGlobal the number of global lattice sites in the y direction
 * @nf the number of features in the layer
 */
CONVERSIONS_SPECIFIER inline float
xPosGlobal(int kGlobal, int xScaleLog2, int nxGlobal, int nyGlobal, int nf) {
   // breaking out variables removes warning from Intel compiler
   const int kxGlobal = kxPos(kGlobal, nxGlobal, nyGlobal, nf);
   const float x0     = xOriginGlobal(xScaleLog2);
   const float dx     = deltaX(xScaleLog2);
   return (x0 + dx * kxGlobal);
}

/**
 * Returns the global y position in physical space
 * @kGlobal the global k index
 * @yScaleLog2 the log2 scale factor for the layer
 *     - e.g. if yScaleLog2 == 1 then dy == 2, if yScaleLog2 == -1 then dy == 1/2
 * @nxGlobal the number of global lattice sites in the x direction
 * @nyGlobal the number of global lattice sites in the y direction
 * @nf the number of features in the layer
 */
CONVERSIONS_SPECIFIER inline float
yPosGlobal(int kGlobal, int yScaleLog2, int nxGlobal, int nyGlobal, int nf) {
   const int kyGlobal = kyPos(kGlobal, nxGlobal, nyGlobal, nf);
   const float y0     = yOriginGlobal(yScaleLog2);
   const float dy     = deltaY(yScaleLog2);
   return (y0 + dy * kyGlobal);
}

//! RETURNS LINEAR INDEX FROM X,Y, AND FEATURE INDEXES
/*!
 * @kx
 * @ky
 * @kf
 * @nx
 * @ny
 * @nf
 * REMARKS:
 *      - This simply says that:
 *      k = ky * (nf*nx) + kx * nf + kf
 *      .
 */
CONVERSIONS_SPECIFIER inline int kIndex(int kx, int ky, int kf, int nx, int ny, int nf) {
   return kf + (kx + ky * nx) * nf;
}

//! RETURNS LINEAR INDEX FROM Batch, X,Y, AND FEATURE INDEXES
CONVERSIONS_SPECIFIER inline int
kIndexBatch(int kb, int kx, int ky, int kf, int nb, int nx, int ny, int nf) {
   return (kb * nx * ny * nf) + (ky * nx * nf) + (kx * nf) + kf;
}

//! Returns stride in feature dimension for linear indexing
/**
 * @loc
  * REMARKS:
 *      - in the linear index space feature index varies first, followed by
 *      X direction index, followed by Y direction index.
 *      - remember that:
 *      k = ky * (nf*nx) + kx * nf + kf
 */
CONVERSIONS_SPECIFIER inline size_t strideF(const PVLayerLoc *loc) { return 1; }

// Version for data structures in extended space (e.g., activity)
CONVERSIONS_SPECIFIER inline size_t strideFExtended(const PVLayerLoc *loc) { return 1; }

//! Returns stride in x dimension for linear indexing
/*!
 * @loc
 *
 * REMARKS:
 *      - in the linear index space feature index varies first, followed by
 *      X direction index, followed by Y direction index.
 *      - remember that:
 *      k = ky * (nf*nx) + kx * nf + kf
 */
CONVERSIONS_SPECIFIER inline size_t strideX(const PVLayerLoc *loc) { return loc->nf; }

// Version for data structures in extended space (e.g., activity)
CONVERSIONS_SPECIFIER inline size_t strideXExtended(const PVLayerLoc *loc) { return loc->nf; }

//! Returns stride in y dimension for linear indexing
/*!
 * @loc
 *
 * REMARKS:
 *      - in the linear index space feature index varies first, followed by
 *      X direction index, followed by Y direction index.
 *      - remember that:
 *      k = ky * (nf*nx) + kx * nf + kf
 */
CONVERSIONS_SPECIFIER inline size_t strideY(const PVLayerLoc *loc) { return loc->nf * loc->nx; }

// Version for data structures in extended space (e.g., activity)
CONVERSIONS_SPECIFIER inline size_t strideYExtended(const PVLayerLoc *loc) {
   return loc->nf * (loc->nx + loc->halo.lt + loc->halo.rt);
}

//! Returns stride in y dimension for linear indexing
/*!
 * @loc
 *
 * REMARKS:
 *      - in the linear index space feature index varies first, followed by
 *      X direction index, followed by Y direction index.
 *      - remember that:
 *      k = ky * (nf*nx) + kx * nf + kf
 */
CONVERSIONS_SPECIFIER inline size_t strideB(const PVLayerLoc *loc) {
   return loc->nf * loc->nx * loc->ny;
}

// Version for data structures in extended space (e.g., activity)
CONVERSIONS_SPECIFIER inline size_t strideBExtended(const PVLayerLoc *loc) {
   return loc->nf * (loc->nx + loc->halo.lt + loc->halo.rt)
          * (loc->ny + loc->halo.up + loc->halo.dn);
}

/**
 * Returns the k direction index of the nearest neighbor in the post-synaptic layer
 *
 * @kzPre the presynaptic index (can be either local or global)
 * @zScaleLog2Pre the log2 scale factor for the presynaptic layer
 * @zScaleLog2Post the log2 scale factor for the postsynaptic layer
 *    - e.g. if zScaleLog2 == 1 then dz == 2, if zScaleLog2 == -1 then dz == 1/2
 *
 *  If the density of the post-synaptic layer increases, the nearby neighbor is
 *  ambiguous and the neuron to the left is chosen.  If the density of the
 *  post-synaptic layer decreases, there is no ambiguity.
 *
 *  presynaptic index should always be in restricted space
 *
 */
CONVERSIONS_SPECIFIER inline int nearby_neighbor(int kzPre, int zLog2ScaleDiff) {
   float a = powf(2.0f, -(float)zLog2ScaleDiff);
   int ia  = (int)a;

   int k0 = (ia < 2) ? 0 : ia / 2 - 1;

   // negative kzPre is different if density of post-synaptic layer decreases
   int k = (a < 1.0f && kzPre < 0) ? kzPre - (int)(1.0f / a) + 1 : kzPre;

   return k0 + (int)(a * k);
}

#define DEPRECATED_FEATURES
#ifdef DEPRECATED_FEATURES
// deprecated
/**
 * Assuming kPre connects to the nearest kPost, return the distance between these two positions
 *    (xPost - xPre) or (yPost - yPre) in units of post-synaptic dx (or dy).
 *
 * @kPre
 * @scale
 */
CONVERSIONS_SPECIFIER inline float deltaPosLayers(int kPre, int scale) {
   if (scale == 0) {
      return 0.0f;
   }
   else if (scale < 0) {
      // post-synaptic layer has smaller size scale
      int s = (int)powf(2.0f, (float)-scale);
      return 0.5f * (float)(1 - s);
   }
   else {
      // post-synaptic layer has larger size scale
      int s = (int)powf(2.0f, (float)scale);
      return 0.5f * (1.0f - (1.0f + 2.0f * (kPre % s)) / s);
   }
}
#endif /* DEPRECATED_FEATURES */

//! RETURNS LINEAR INDEX IN THE EXTENDED SPACE FROM INDICES IN RESTRICTED SPACE
/*!
 * @k the k index in restricted space
 * @nx the size in x of restricted space
 * @ny the size in y of restricted space
 * @nf the size in f of restricted space
 * @lt the width of the left margin
 * @rt the width of the right margin
 * @dn the width of the bottom margin
 * @up the width of, you guessed it, the top margin
 *
 * REMARKS:
 *   - the linear indexing of neurons is done by varying first along these directions:
 *   feature direction, X direction, Y direction.
 *   - for given indices kf,kx,ky, the linear index k restricted is given by:
 *     k = ky*(nf*nx) + kx*nf + kf
 *   - kx is the X direction index in restricted space
 *   - ky is the Y direction index in restricted space
 *   .
 */
CONVERSIONS_SPECIFIER inline int
kIndexExtended(int k, int nx, int ny, int nf, int lt, int rt, int dn, int up) {
   const int kx_ex = lt + kxPos(k, nx, ny, nf);
   const int ky_ex = up + kyPos(k, nx, ny, nf);
   const int kf    = featureIndex(k, nx, ny, nf);
   return kIndex(kx_ex, ky_ex, kf, nx + lt + rt, ny + dn + up, nf);
}

//! RETURNS LINEAR INDEX IN THE EXTENDED SPACE FROM INDICES IN RESTRICTED SPACE
/*!
 * @k the k index in restricted space
 * @nx the size in x of restricted space
 * @ny the size in y of restricted space
 * @nf the size in f of restricted space
 * @nb the size of batch
 * @lt the width of the left margin
 * @rt the width of the right margin
 * @dn the width of the bottom margin
 * @up the width of, you guessed it, the top margin
 *
 * REMARKS:
 *   - the linear indexing of neurons is done by varying first along these directions:
 *   feature direction, X direction, Y direction.
 *   - for given indices kf,kx,ky, the linear index k restricted is given by:
 *     k = ky*(nf*nx) + kx*nf + kf
 *   - kx is the X direction index in restricted space
 *   - ky is the Y direction index in restricted space
 *   .
 */
CONVERSIONS_SPECIFIER inline int
kIndexExtendedBatch(int k, int nb, int nx, int ny, int nf, int lt, int rt, int dn, int up) {
   const int kx_ex = lt + kxPos(k, nx, ny, nf);
   const int ky_ex = up + kyPos(k, nx, ny, nf);
   const int kf    = featureIndex(k, nx, ny, nf);
   const int kb    = batchIndex(k, nb, nx, ny, nf);
   return kIndexBatch(kb, kx_ex, ky_ex, kf, nb, nx + lt + rt, ny + dn + up, nf);
}

/*!
 * Returns the k linear index in restricted space from the kex index
 * in extended space or # < 0 if k_ex is in border region
 * @k_ex the linear k index in extended space
 * @nx the size in x of restricted space
 * @ny the size in y of restricted space
 * @nf the size in f of restricted space
 * @lt the width of the left margin
 * @rt the width of the right margin
 * @dn the width of the bottom margin
 * @up the width of the top margin
 *
 * REMARKS:
 *   - the linear indexing of neurons is done by varying first along these directions:
 *   feature direction, X direction, Y direction.
 *   - for given indices kf,kx,ky, the linear index k restricted is given by:
 *     k = ky*(nf*nx) + kx*nf + kf
 *   - kx is the X direction index in restricted space
 *   - ky is the Y direction index in restricted space
 *   .
 */
CONVERSIONS_SPECIFIER inline int
kIndexRestricted(int k_ex, int nx, int ny, int nf, int lt, int rt, int dn, int up) {
   int kx, ky, kf;

   const int nx_ex = nx + lt + rt;
   const int ny_ex = ny + dn + up;

   kx = kxPos(k_ex, nx_ex, ny_ex, nf) - lt;
   if (kx < 0 || kx >= nx)
      return -1;

   ky = kyPos(k_ex, nx_ex, ny_ex, nf) - up;
   if (ky < 0 || ky >= ny)
      return -1;

   kf = featureIndex(k_ex, nx_ex, ny_ex, nf);
   return kIndex(kx, ky, kf, nx, ny, nf);
}

/**
 * @kl
 * @loc
 * @nf
 */
// TODO - put back in nx,ny,... so that it will vectorize with vector of kl's
//
// WARNING - If changed, copy changes to the tests/test_kg.c and
//           tests/test_extend_border.c files. These tests run a
//           function equivalent to the mpi version of
//           globalIndexFromLocal but without using MPI.
CONVERSIONS_SPECIFIER inline int globalIndexFromLocal(int kl, const PVLayerLoc loc) {
#ifdef PV_USE_MPI
   int kxg = loc.kx0 + kxPos(kl, loc.nx, loc.ny, loc.nf);
   int kyg = loc.ky0 + kyPos(kl, loc.nx, loc.ny, loc.nf);
   int kf  = featureIndex(kl, loc.nx, loc.ny, loc.nf);
   return kIndex(kxg, kyg, kf, loc.nxGlobal, loc.nyGlobal, loc.nf);
#else
   return kl;
#endif // PV_USE_MPI
}

CONVERSIONS_SPECIFIER inline int localIndexFromGlobal(int kGlobal, const PVLayerLoc loc) {
#ifdef PV_USE_MPI
   int kxGlobal = kxPos(kGlobal, loc.nxGlobal, loc.nyGlobal, loc.nf);
   int kyGlobal = kyPos(kGlobal, loc.nxGlobal, loc.nyGlobal, loc.nf);
   int kf       = featureIndex(kGlobal, loc.nxGlobal, loc.nyGlobal, loc.nf);
   int kxLocal  = kxGlobal - loc.kx0;
   int kyLocal  = kyGlobal - loc.ky0;
   return kIndex(kxLocal, kyLocal, kf, loc.nx, loc.ny, loc.nf);
#else
   return kGlobal;
#endif // PV_USE_MPI
}

/**
 * Gives the size of the unit cell (either x or y dimension) of a patch for a HyPerConn
 * whose pre- and post-layers have the given dimensions.
 */
CONVERSIONS_SPECIFIER inline int zUnitCellSize(int preZSize, int postZSize) {
   return (preZSize > postZSize) ? preZSize / postZSize : 1;
}

/**
 * @x
 */
CONVERSIONS_SPECIFIER inline float sign(float x) { return (x < 0.0f) ? -1.0f : 1.0f; }

/**
 * Returns difference between two numbers assuming periodic boundary conditions.
 * IMPORTANT NOTE - assumes abs(x2-x1) < 2*max and max > 0
 * @x2 first number
 * @x2 second number
 * @max maximum difference
 */
//#pragma FTT elemental, vectorize
CONVERSIONS_SPECIFIER inline float deltaWithPBC(float x1, float x2, float max) {
   float dx     = x2 - x1;
   float abs_dx = fabsf(dx);

   // Apply periodic boundary conditions
   dx = abs_dx > max ? sign(dx) * (abs_dx - 2.0f * max) : dx;

   return dx;
}

/**
 * Return global k index from x,y position information
 * @kf the feature index (not the k index as other routines use)
 * @x
 * @y
 * @x0
 * @y0
 * @dx
 * @dy
 * @nx
 * @ny
 * @nf
 */
//#pragma FTT elemental, vectorize
CONVERSIONS_SPECIFIER inline int globalIndex(
      int kf,
      float x,
      float y,
      float x0,
      float y0,
      float dx,
      float dy,
      int nx,
      int ny,
      int nf) {
   int kx = (int)nearbyintf((x - x0) / dx - 0.5f);
   int ky = (int)nearbyintf((y - y0) / dy - 0.5f);
   return kIndex(kx, ky, kf, nx, ny, nf);
}

// Converts an index from one layer to the other in the extended space
// Warning: function will return center point in a one to many conversion
// Conversion in feature space does not exist, output will be first feature
// If outside the area of out layer, will move to the clostest avaliable position in out layer
CONVERSIONS_SPECIFIER inline int
layerIndexExt(int kPreExt, const PVLayerLoc *inLoc, const PVLayerLoc *outLoc) {
   // Calculate scale factor based on restricted
   float scaleFactorX = (float)outLoc->nxGlobal / inLoc->nxGlobal;
   float scaleFactorY = (float)outLoc->nyGlobal / inLoc->nyGlobal;
   // Calculate x and y in extended space
   int kPreX =
         kxPos(kPreExt,
               inLoc->nx + inLoc->halo.lt + inLoc->halo.rt,
               inLoc->ny + inLoc->halo.dn + inLoc->halo.up,
               inLoc->nf);
   int kPreY =
         kyPos(kPreExt,
               inLoc->nx + inLoc->halo.lt + inLoc->halo.rt,
               inLoc->ny + inLoc->halo.dn + inLoc->halo.up,
               inLoc->nf);
   // Subtract margin to set 0 to the beginning of the restricted space
   kPreX -= inLoc->halo.lt;
   kPreY -= inLoc->halo.up;
   int kPostX, kPostY, half;
   // If one to many, scale factor is greater than 1
   if (scaleFactorX > 1) {
      half   = floor(scaleFactorX / 2);
      kPostX = kPreX * scaleFactorX + half;
   }
   else {
      kPostX = floor(kPreX * scaleFactorX);
   }
   if (scaleFactorY > 1) {
      half   = floor(scaleFactorY / 2);
      kPostY = kPreY * scaleFactorY + half;
   }
   else {
      kPostY = floor(kPreY * scaleFactorY);
   }

   // Change back to ext points
   kPostX += outLoc->halo.lt;
   kPostY += outLoc->halo.up;

   // If outside of out layer margins, shrink
   // Left margin
   if (kPostX < 0) {
      kPostX = 0;
   }
   // Right Margin
   else if (kPostX >= outLoc->nx + outLoc->halo.lt + outLoc->halo.rt) {
      kPostX = outLoc->nx + outLoc->halo.dn + outLoc->halo.up - 1;
   }
   // Top margin
   if (kPostY < 0) {
      kPostY = 0;
   }
   // Bottom Margin
   else if (kPostY >= outLoc->ny + outLoc->halo.lt + outLoc->halo.rt) {
      kPostY = outLoc->ny + outLoc->halo.dn + outLoc->halo.up - 1;
   }
   // Change back to index
   // Using feature of 0
   return kIndex(
         kPostX,
         kPostY,
         0,
         outLoc->nx + outLoc->halo.lt + outLoc->halo.rt,
         outLoc->ny + outLoc->halo.dn + outLoc->halo.up,
         outLoc->nf);
}

// Converts an index from one layer to the other in the restricted space
// Warning: function will return center point in a one to many conversion
// Conversion in feature space does not exist, output will be first feature
CONVERSIONS_SPECIFIER inline int
layerIndexRes(int kPreRes, const PVLayerLoc *inLoc, const PVLayerLoc *outLoc) {
   // Call with extended index
   int kPreExt = kIndexExtended(
         kPreRes,
         inLoc->nx,
         inLoc->ny,
         inLoc->nf,
         inLoc->halo.lt,
         inLoc->halo.rt,
         inLoc->halo.dn,
         inLoc->halo.up);
   return layerIndexExt(kPreExt, inLoc, outLoc);
}

/**
 * Returns 1 if the given extended index is in the border region, and 0 if it is in the restricted
 * space.
 */
CONVERSIONS_SPECIFIER inline int
extendedIndexInBorderRegion(int extK, int nx, int ny, int nf, int lt, int rt, int dn, int up) {
   int x = kxPos(extK, nx + lt + rt, ny + dn + up, nf);
   int y = kyPos(extK, nx + lt + rt, ny + dn + up, nf);
   return (x < lt) | (x >= nx + lt) | (y < up) | (y >= ny + up);
   // Which is better: bitwise-or or logical-or?
}

// Converts a local ext index into a global res index
// Returns -1 if localExtK is in extended space
CONVERSIONS_SPECIFIER inline int localExtToGlobalRes(int localExtK, const PVLayerLoc *loc) {
   // Change local ext indicies to global res index
   int localExtX =
         kxPos(localExtK,
               loc->nx + loc->halo.lt + loc->halo.rt,
               loc->ny + loc->halo.up + loc->halo.dn,
               loc->nf);
   int localExtY =
         kyPos(localExtK,
               loc->nx + loc->halo.lt + loc->halo.rt,
               loc->ny + loc->halo.up + loc->halo.dn,
               loc->nf);
   int localF = featureIndex(
         localExtK,
         loc->nx + loc->halo.lt + loc->halo.rt,
         loc->ny + loc->halo.up + loc->halo.dn,
         loc->nf);

   // Check if restricted
   if (localExtX < loc->halo.lt || localExtX >= loc->nx + loc->halo.lt || localExtY < loc->halo.up
       || localExtY >= loc->ny + loc->halo.up) {
      return -1;
   }

   // Change ext to res
   int globalResX = localExtX - loc->halo.lt + loc->kx0;
   int globalResY = localExtY - loc->halo.up + loc->ky0;

   // Get final globalResK
   int globalResK = kIndex(globalResX, globalResY, localF, loc->nxGlobal, loc->nyGlobal, loc->nf);
   return globalResK;
}

/**
 * @x0
 * @x
 * @sigma
 * @max
 */
//#pragma FTT elemental, vectorize
CONVERSIONS_SPECIFIER inline float gaussianWeight(float x0, float x, float sigma, float max) {
   float dx = deltaWithPBC(x0, x, max);
   return expf(-0.5f * dx * dx / (sigma * sigma));
}

CONVERSIONS_SPECIFIER inline int
rankFromRowAndColumn(int row, int column, int numRows, int numColumns) {
   bool inbounds = row >= 0 and row < numRows and column >= 0 and column < numColumns;
   return inbounds ? row * numColumns + column : -1;
}

CONVERSIONS_SPECIFIER inline int rankFromRowColumnBatch(
      int row,
      int column,
      int batch,
      int numRows,
      int numColumns,
      int batchWidth) {
   bool inbounds = row >= 0 and row < numRows and column >= 0 and column < numColumns;
   inbounds &= batch >= 0 and batch < batchWidth;
   return inbounds ? column + numColumns * (row + numRows * batch) : -1;
}

CONVERSIONS_SPECIFIER inline int
globalToLocalRank(int rank, int batchWidth, int numRows, int numColumns) {
   // This line will not do anything if the parameter rank is a localRank
   int localRank = rank % (numRows * numColumns);
   return localRank;
}

CONVERSIONS_SPECIFIER inline int rowFromRank(int rank, int numRows, int numColumns) {
   int row = rank / numColumns;
   if (row < 0 || row >= numRows)
      row = -1;
   return row;
}

CONVERSIONS_SPECIFIER inline int columnFromRank(int rank, int numRows, int numColumns) {
   int col = rank % numColumns;
   if (col < 0 || col >= numColumns)
      col = -1;
   return col;
}

CONVERSIONS_SPECIFIER inline int
batchFromRank(int rank, int batchWidth, int numRows, int numColumns) {
   int col = rank / (numRows * numColumns);
   if (col < 0 || col >= batchWidth)
      col = -1;
   return col;
}

/**
 * For a convolution-based connection between two layers, computes the
 * margin width the presynaptic layer must have, given the size of the
 * presynaptic and postsynaptic layers, and the patch size (the number of
 * postsynaptic neurons each presynaptic neuron connects to).
 * If nPre and nPost are not the same, the larger must be an 2^k times the
 * smaller for some positive integer k.
 * If nPre == nPost, patchSize must be odd.
 * If nPre > nPost (many-to-one), any patchSize is permissible.
 * If nPost > nPre (one-to-many), patchSize must be a multiple of (nPost/nPre).
 */
CONVERSIONS_SPECIFIER inline int requiredConvolveMargin(int nPre, int nPost, int patchSize) {
   int margin = 0;
   if (nPre == nPost) {
      assert(patchSize % 2 == 1);
      margin = (patchSize - 1) / 2;
   }
   else if (nPre > nPost) { // many-to-one
      assert(nPre % nPost == 0);
      int densityRatio = nPre / nPost;
      assert(densityRatio % 2 == 0);
      assert(pow(2.0, nearbyint(log2((double)densityRatio))) == (double)densityRatio);
      margin = (patchSize - 1) * densityRatio / 2;
   }
   else {
      assert(nPre < nPost); // one-to-many
      assert(nPost % nPre == 0);
      int densityRatio = nPost / nPre;
      assert(densityRatio % 2 == 0);
      assert(pow(2.0, nearbyint(log2((double)densityRatio))) == (double)densityRatio);
      assert(patchSize % densityRatio == 0);
      int numCells = patchSize / densityRatio;
      margin       = numCells / 2;
      // integer division is correct, no matter whether numCells is even or odd
   }
   return margin;
}

} // end namespace PV

#endif /* CONVERSIONS_H_ */
