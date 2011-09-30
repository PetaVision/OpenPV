/*
 * conversions.h
 *
 *  Created on: Jan 7, 2010
 *      Author: rasmussn
 */

#ifndef CONVERSIONS_H_
#define CONVERSIONS_H_

/* define this variable if code is to be run (or transformed to be run) on a Cell processor */
//#define CELL_BE

#include "../include/pv_types.h"

#include <math.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C"
{
#endif // __cplusplus

   int dist2NearestCell(int kzPre, int log2ScalePre, int log2ScalePost, float * distPre,
      float * distPost);

   int zPatchHead(int kzPre, int nzPatch, int zScaleLog2Pre, int zScaleLog2Post);

   int posPatchHead(const int kPre, const int xScaleLog2Pre,
         const int yScaleLog2Pre, const PVLayerLoc locPre, float * xPreGlobal,
         float * yPreGlobal, const int xScaleLog2Post, const int yScaleLog2Post,
         const PVLayerLoc locPost, const PVPatch * wp, float * xPatchHeadGlobal,
         float * yPatchHeadGlobal);

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
static inline int featureIndex(int k, int nx, int ny, int nf)
{
   return k % nf;
}

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
static inline int kxPos(int k, int nx, int ny, int nf)
{
   return (k/nf) % nx;
}

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
static inline int kyPos(int k, int nx, int ny, int nf)
{
   return k / (nx*nf);
}

/**
 * Returns the x dimension scale length for the layer in retinatopic units
 * where dx == 1
 * @xScaleLog2 the log2 scale factor for the layer
 *     - e.g. if xScaleLog2 == 1 then dx == 2, if xScaleLog2 == -1 then dx == 1/2
 */
static inline float deltaX(int xScaleLog2)
{
   return powf(2.0f, (float) xScaleLog2);
}

/**
 * Returns the y dimension scale length for the layer in retinatopic units
 * where dy == 1
 * @yScaleLog2 the log2 scale factor for the layer
 *     - e.g. if yScaleLog2 == 1 then dy == 2, if yScaleLog2 == -1 then dy == 1/2
 */
static inline float deltaY(int yScaleLog2)
{
   return powf(2.0f, (float) yScaleLog2);
}

/**
 * Returns the _global_ x origin in retinatopic units where dx == 1
 * @xScaleLog2 the log2 scale factor for the layer
 *     - e.g. if xScaleLog2 == 1 then dx == 2, if xScaleLog2 == -1 then dx == 1/2
 */
static inline float xOriginGlobal(int xScaleLog2)
{
   return 0.5f * deltaX(xScaleLog2);
}

/**
 * Returns the _global_ y origin in retinatopic units where dy == 1
 * @yScaleLog2 the log2 scale factor for the layer
 *     - e.g. if yScaleLog2 == 1 then dy == 2, if yScaleLog2 == -1 then dy == 1/2
 */
static inline float yOriginGlobal(int yScaleLog2)
{
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
static inline float xPosGlobal(int kGlobal, int xScaleLog2,
                               int nxGlobal, int nyGlobal, int nf)
{
   // breaking out variables removes warning from Intel compiler
   const int kxGlobal = kxPos(kGlobal, nxGlobal, nyGlobal, nf);
   const float x0 = xOriginGlobal(xScaleLog2);
   const float dx = deltaX(xScaleLog2);
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
static inline float yPosGlobal(int kGlobal, int yScaleLog2,
                               int nxGlobal, int nyGlobal, int nf)
{
   const int kyGlobal = kyPos(kGlobal, nxGlobal, nyGlobal, nf);
   const float y0 = yOriginGlobal(yScaleLog2);
   const float dy = deltaY(yScaleLog2);
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
static inline int kIndex(int kx, int ky, int kf, int nx, int ny, int nf)
{
   return kf + (kx + ky * nx) * nf;
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
static inline size_t strideF(const PVLayerLoc * loc)
{
   return 1;
}

// Version for data structures in extended space (e.g., activity)
static inline size_t strideFExtended(const PVLayerLoc * loc)
{
   return 1;
}

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
static inline size_t strideX(const PVLayerLoc * loc)
{
   return loc->nf;
}

// Version for data structures in extended space (e.g., activity)
static inline size_t strideXExtended(const PVLayerLoc * loc)
{
   return loc->nf;
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
static inline size_t strideY(const PVLayerLoc * loc)
{
   return loc->nf*loc->nx;
}

// Version for data structures in extended space (e.g., activity)
static inline size_t strideYExtended(const PVLayerLoc * loc)
{
   return loc->nf*(loc->nx + loc->halo.lt + loc->halo.rt);
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
static inline int nearby_neighbor(int kzPre, int zScaleLog2Pre, int zScaleLog2Post)
{
   float a = powf(2.0f, (float) (zScaleLog2Pre - zScaleLog2Post));
   int ia = (int) a;

   int k0 = (ia < 2) ? 0 : ia/2 - 1;

   // negative kzPre is different if density of post-synaptic layer decreases
   int k  = (a < 1.0f && kzPre < 0) ? kzPre - (int) (1.0f/a) + 1 : kzPre;

   return k0 + (int) (a * k);
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
static inline float deltaPosLayers(int kPre, int scale)
{
   if (scale == 0) {
      return 0.0f;
   }
   else if (scale < 0) {
      // post-synaptic layer has smaller size scale
      int s = (int) powf(2.0f, (float) -scale);
      return 0.5f * (float) (1 - s);
   }
   else {
      // post-synaptic layer has larger size scale
      int s = (int) powf(2.0f, (float) scale);
      return 0.5f * (1.0f - (1.0f + 2.0f * (kPre%s)) / s);
   }
}
#endif /* DEPRECATED_FEATURES */

//! RETURNS LINEAR INDEX IN THE EXTENDED SPACE FROM INDICES IN RESTRICTED SPACE
/*!
 * @k the k index in restricted space
 * @nx the size in x of restricted space
 * @ny the size in y of restricted space
 * @nf the size in f of restricted space
 * @nb the width of the margin
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
static inline int kIndexExtended(int k, int nx, int ny, int nf, int nb)
{
   const int kx_ex = nb + kxPos(k, nx, ny, nf);
   const int ky_ex = nb + kyPos(k, nx, ny, nf);
   const int kf = featureIndex(k, nx, ny, nf);
   return kIndex(kx_ex, ky_ex, kf, nx + 2*nb, ny + 2*nb, nf);
}

/*!
 * Returns the k linear index in restricted space from the kex index
 * in extended space or # < 0 if k_ex is in border region
 * @k_ex the linear k index in extended space
 * @nx the size in x of restricted space
 * @ny the size in y of restricted space
 * @nf the size in f of restricted space
 * @nb the width of the margin
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
static inline int kIndexRestricted(int k_ex, int nx, int ny, int nf, int nb)
{
   int kx, ky, kf;

   const int nx_ex = nx + 2*nb;
   const int ny_ex = ny + 2*nb;

   kx = kxPos(k_ex, nx_ex, ny_ex, nf) - nb;
   if (kx < 0 || kx >= nx) return -1;

   ky = kyPos(k_ex, nx_ex, ny_ex, nf) - nb;
   if (ky < 0 || ky >= ny) return -1;

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
static inline int globalIndexFromLocal(int kl, PVLayerLoc loc)
{
#ifdef PV_USE_MPI
   int kxg = loc.kx0 + kxPos(kl, loc.nx, loc.ny, loc.nf);
   int kyg = loc.ky0 + kyPos(kl, loc.nx, loc.ny, loc.nf);
   int  kf = featureIndex(kl, loc.nx, loc.ny, loc.nf);
   return kIndex(kxg, kyg, kf, loc.nxGlobal, loc.nyGlobal, loc.nf);
#else
   return kl;
#endif // PV_USE_MPI
}


static inline int localIndexFromGlobal(int kGlobal, PVLayerLoc loc)
{
#ifdef PV_USE_MPI
   int kxGlobal = kxPos(kGlobal, loc.nxGlobal, loc.nxGlobal, loc.nf);
   int kyGlobal = kyPos(kGlobal, loc.nxGlobal, loc.nxGlobal, loc.nf);
   int kf = featureIndex(kGlobal, loc.nxGlobal, loc.nxGlobal, loc.nf);
   int kxLocal = kxGlobal - loc.kx0;
   int kyLocal = kyGlobal - loc.ky0;
   return kIndex(kxLocal, kyLocal, kf, loc.nx, loc.ny, loc.nf);
#else
   return kGlobal;
#endif // PV_USE_MPI
}

/**
 * @x
 */
static inline float sign(float x)
{
    return (x < 0.0f) ? -1.0f : 1.0f;
}

/**
 * Returns difference between two numbers assuming periodic boundary conditions.
 * IMPORTANT NOTE - assumes abs(x2-x1) < 2*max and max > 0
 * @x2 first number
 * @x2 second number
 * @max maximum difference
 */
//#pragma FTT elemental, vectorize
static inline float deltaWithPBC(float x1, float x2, float max)
{
    float dx = x2 - x1;
    float abs_dx = fabsf(dx);

    // Apply periodic boundary conditions
    dx = abs_dx > max ? sign(dx) * (abs_dx - 2.0f*max) : dx;

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
static inline int globalIndex(int kf, float x, float y, float x0, float y0,
                              float dx, float dy, int nx, int ny, int nf)
{
   int kx = (int) nearbyintf((x - x0)/dx - 0.5f);
   int ky = (int) nearbyintf((y - y0)/dy - 0.5f);
   return kIndex(kx, ky, kf, nx, ny, nf);
}

/**
 * @x0
 * @x
 * @sigma
 * @max
 */
//#pragma FTT elemental, vectorize
static inline float gaussianWeight(float x0, float x, float sigma, float max)
{
   float dx = deltaWithPBC(x0, x, max);
   return expf(-0.5f * dx * dx / (sigma * sigma));
}

#ifdef __cplusplus
}
#endif // __cplusplus

#endif /* CONVERSIONS_H_ */
