/*
 * conversions.hpp
 *
 *  Created on: Jan 7, 2010
 *      Author: rasmussn
 */

#ifndef CONVERSIONS_HPP_
#define CONVERSIONS_HPP_

#include "structures/PVLayerLoc.hpp"
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
      float scaleFactor       = std::pow(2.0f, static_cast<float>(zLog2ScaleDiff));
      float kzPreToPostCoords = ((float)kzPre - 0.5f * (scaleFactor - 1.0f)) / scaleFactor;
      float kzPost            = round(kzPreToPostCoords);
      *distPost               = kzPost - kzPreToPostCoords;
      *distPre                = *distPost * scaleFactor;
      return (int)kzPost;
   }
   else {
      assert(zLog2ScaleDiff < 0);
      // one-to-many case
      float scaleFactor = std::pow(2.0f, static_cast<float>(-zLog2ScaleDiff));
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
      float tstride         = std::pow(2.0f, static_cast<float>(zLog2ScaleDiff));
      float halfWidth       = 0.5f * static_cast<float>(nzPatch - 1);
      float zPreInPostSpace = (static_cast<float>(kzPre) + 0.5f) / tstride;
      return (int)std::floor(zPreInPostSpace - halfWidth);
   }
   else {
      assert(zLog2ScaleDiff < 0);
      // one-to-many case
      int stride = static_cast<int>(std::pow(2, -zLog2ScaleDiff));
      return kzPre * stride - (nzPatch - stride) / 2;
      // A note regarding integer arithmetic. stride must be even here, and the typical use case
      // is that nzPatch is an integer multiple of stride; then there is no truncation from
      // integer division. If nzPatch is odd, the result is the same as if nzPatch-1 were given.
   }
}

/*
 * The following functions are simple, static inline functions.
 *
 * Notation:
 *
 *   scaleLog2 - absolute distance scale (between neurons) of a layer relative to retina
 *     - e.g. if xScaleLog2 == 1 then dx == 2, if xScaleLog2 == -1 then dx == 1/2
 *
 */

/** RETURNS FEATURE INDEX FROM LINEAR INDEX
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
CONVERSIONS_SPECIFIER inline int featureIndex(long k, int nx, int ny, int nf) {
   return static_cast<int>(k % nf);
}

/** RETURNS X INDEX FROM LINEAR INDEX
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
CONVERSIONS_SPECIFIER inline int kxPos(long k, int nx, int ny, int nf) {
   return static_cast<int>((k / nf) % nx);
}

/** RETURNS Y INDEX FROM LINEAR INDEX
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
CONVERSIONS_SPECIFIER inline int kyPos(long k, int nx, int ny, int nf) {
   return static_cast<int>(k / (nx * nf)) % ny;
}

/** RETURNS B INDEX FROM LINEAR INDEX
 * Return the position kb for the given k index into an nb-nx-by-ny-by-nf 4-D array
 * @k the k index (can be either global or local depending on if nx,ny,nb are global or local)
 * @nb the number of batch elements
 * @nx the number of neurons in the x direction
 * @ny the number of neurons in the y direction
 * @nf the number of neurons in the feature direction
 */
CONVERSIONS_SPECIFIER inline int batchIndex(long k, int nb, int nx, int ny, int nf) {
   return static_cast<int>(k / ((long)nx * (long)ny * (long)nf));
}

/**
 * Returns the x dimension scale length for the layer in retinatopic units
 * where dx == 1
 * @xScaleLog2 the log2 scale factor for the layer
 *     - e.g. if xScaleLog2 == 1 then dx == 2, if xScaleLog2 == -1 then dx == 1/2
 */
CONVERSIONS_SPECIFIER inline float deltaX(int xScaleLog2) {
   return std::pow(2.0f, (float)xScaleLog2);
}

/**
 * Returns the y dimension scale length for the layer in retinatopic units
 * where dy == 1
 * @yScaleLog2 the log2 scale factor for the layer
 *     - e.g. if yScaleLog2 == 1 then dy == 2, if yScaleLog2 == -1 then dy == 1/2
 */
CONVERSIONS_SPECIFIER inline float deltaY(int yScaleLog2) {
   return std::pow(2.0f, (float)yScaleLog2);
}

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
xPosGlobal(long kGlobal, int xScaleLog2, int nxGlobal, int nyGlobal, int nf) {
   // breaking out variables removes warning from Intel compiler
   const int kxGlobal = kxPos(kGlobal, nxGlobal, nyGlobal, nf);
   const float x0     = xOriginGlobal(xScaleLog2);
   const float dx     = deltaX(xScaleLog2);
   return (x0 + dx * static_cast<float>(kxGlobal));
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
yPosGlobal(long kGlobal, int yScaleLog2, int nxGlobal, int nyGlobal, int nf) {
   const int kyGlobal = kyPos(kGlobal, nxGlobal, nyGlobal, nf);
   const float y0     = yOriginGlobal(yScaleLog2);
   const float dy     = deltaY(yScaleLog2);
   return (y0 + dy * static_cast<float>(kyGlobal));
}

/** RETURNS LINEAR INDEX FROM X,Y, AND FEATURE INDEXES
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
CONVERSIONS_SPECIFIER inline long kIndex(int kx, int ky, int kf, int nx, int ny, int nf) {
   return kf + (kx + ky * (long)nx) * (long)nf;
}

//! RETURNS LINEAR INDEX INTO 4-D FROM Batch, X,Y, AND FEATURE INDEXES
CONVERSIONS_SPECIFIER inline long
kIndexBatch(int kb, int kx, int ky, int kf, int nb, int nx, int ny, int nf) {
   long nxL = static_cast<long>(nx);
   long nyL = static_cast<long>(ny);
   long nfL = static_cast<long>(nf);
   return (kb * nxL * nyL * nfL) + (ky * nxL * nfL) + (kx * nfL) + kf;
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
CONVERSIONS_SPECIFIER inline long strideF(const PVLayerLoc *loc) { return 1L; }

// Version for data structures in extended space (e.g., activity)
CONVERSIONS_SPECIFIER inline long strideFExtended(const PVLayerLoc *loc) { return 1L; }

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
CONVERSIONS_SPECIFIER inline long strideX(const PVLayerLoc *loc) { return loc->nf; }

// Version for data structures in extended space (e.g., activity)
CONVERSIONS_SPECIFIER inline long strideXExtended(const PVLayerLoc *loc) { return loc->nf; }

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
CONVERSIONS_SPECIFIER inline long strideY(const PVLayerLoc *loc) {
   return (long)loc->nf * (long)loc->nx;
}

// Version for data structures in extended space (e.g., activity)
CONVERSIONS_SPECIFIER inline long strideYExtended(const PVLayerLoc *loc) {
   return (long)loc->nf * (long)(loc->nx + loc->halo.lt + loc->halo.rt);
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
   return (long)loc->nf * (long)loc->nx * (long)loc->ny;
}

// Version for data structures in extended space (e.g., activity)
CONVERSIONS_SPECIFIER inline size_t strideBExtended(const PVLayerLoc *loc) {
   return (long)loc->nf *
          (long)(loc->nx + loc->halo.lt + loc->halo.rt) *
          (long)(loc->ny + loc->halo.up + loc->halo.dn);
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
   double a = std::pow(2, -zLog2ScaleDiff);
   int ia  = (int)a;

   int k0 = (ia < 2) ? 0 : ia / 2 - 1;

   // negative kzPre is different if density of post-synaptic layer decreases
   int k = (a < 1.0 && kzPre < 0) ? kzPre - (int)(1.0 / a) + 1 : kzPre;

   return k0 + (int)(a * static_cast<double>(k));
}

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
CONVERSIONS_SPECIFIER inline long
kIndexExtended(long k, int nx, int ny, int nf, int lt, int rt, int dn, int up) {
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
CONVERSIONS_SPECIFIER inline long
kIndexExtendedBatch(long kRes, int nb, int nx, int ny, int nf, int lt, int rt, int dn, int up) {
   const int kx_ex = lt + kxPos(kRes, nx, ny, nf);
   const int ky_ex = up + kyPos(kRes, nx, ny, nf);
   const int kf    = featureIndex(kRes, nx, ny, nf);
   const int kb    = batchIndex(kRes, nb, nx, ny, nf);
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
CONVERSIONS_SPECIFIER inline long
kIndexRestricted(long k_ex, int nx, int ny, int nf, int lt, int rt, int dn, int up) {
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
CONVERSIONS_SPECIFIER inline long globalIndexFromLocal(long kl, const PVLayerLoc loc) {
#ifdef PV_USE_MPI
   int kxg = (loc.bcast ? 0 : loc.kx0) + kxPos(kl, loc.nx, loc.ny, loc.nf);
   int kyg = (loc.bcast ? 0 : loc.ky0) + kyPos(kl, loc.nx, loc.ny, loc.nf);
   int kf  = featureIndex(kl, loc.nx, loc.ny, loc.nf);
   return kIndex(kxg, kyg, kf, loc.nxGlobal, loc.nyGlobal, loc.nf);
#else
   return kl;
#endif // PV_USE_MPI
}

CONVERSIONS_SPECIFIER inline long localIndexFromGlobal(long kGlobal, const PVLayerLoc loc) {
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
CONVERSIONS_SPECIFIER inline long globalIndex(
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
CONVERSIONS_SPECIFIER inline long
layerIndexExt(long kPreExt, const PVLayerLoc *inLoc, const PVLayerLoc *outLoc) {
   // Calculate scale factor based on restricted
   float scaleFactorX = static_cast<float>(outLoc->nxGlobal) / static_cast<float>(inLoc->nxGlobal);
   float scaleFactorY = static_cast<float>(outLoc->nyGlobal) / static_cast<float>(inLoc->nyGlobal);
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
   int kPostX, kPostY;
   // If one to many, scale factor is greater than 1
   if (scaleFactorX > 1.0f) {
      int half = static_cast<int>(std::floor(scaleFactorX / 2.0f));
      kPostX   = static_cast<int>(static_cast<float>(kPreX) * scaleFactorX) + half;
   }
   else {
      kPostX = static_cast<int>(std::floor(static_cast<float>(kPreX) * scaleFactorX));
   }
   if (scaleFactorY > 1.0f) {
      int half = (int)std::floor(scaleFactorY / 2.0f);
      kPostY   = static_cast<int>(static_cast<float>(kPreY) * scaleFactorY) + half;
   }
   else {
      kPostY = static_cast<int>(std::floor(static_cast<float>(kPreY) * scaleFactorY));
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
CONVERSIONS_SPECIFIER inline long
layerIndexRes(long kPreRes, const PVLayerLoc *inLoc, const PVLayerLoc *outLoc) {
   // Call with extended index
   long kPreExt = kIndexExtended(
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
extendedIndexInBorderRegion(long extK, int nx, int ny, int nf, int lt, int rt, int dn, int up) {
   int x = kxPos(extK, nx + lt + rt, ny + dn + up, nf);
   int y = kyPos(extK, nx + lt + rt, ny + dn + up, nf);
   return (x < lt) | (x >= nx + lt) | (y < up) | (y >= ny + up);
   // Which is better: bitwise-or or logical-or?
}

// Converts a local ext index into a global res index
// Returns -1 if localExtK is in extended space
CONVERSIONS_SPECIFIER inline long localExtToGlobalRes(long localExtK, const PVLayerLoc *loc) {
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

   // Get final global restricted index
   return kIndex(globalResX, globalResY, localF, loc->nxGlobal, loc->nyGlobal, loc->nf);
}

/**
 * @x0
 * @x
 * @sigma
 * @max
 */
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

} // end namespace PV

#endif // CONVERSIONS_HPP_
