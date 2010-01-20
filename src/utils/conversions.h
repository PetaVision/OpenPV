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
#endif

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
 *   scaleLog2 - absolute scale of a layer relative to retina
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
   return powf(2.0, (float) xScaleLog2);
}

/**
 * Returns the y dimension scale length for the layer in retinatopic units
 * where dy == 1
 * @yScaleLog2 the log2 scale factor for the layer
 *     - e.g. if yScaleLog2 == 1 then dy == 2, if yScaleLog2 == -1 then dy == 1/2
 */
static inline float deltaY(int yScaleLog2)
{
   return powf(2.0, (float) yScaleLog2);
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
   return (int)kf + ((int)kx + (int)ky * (int)nx) * (int)nf;
}

//! RETURNS STRIDE IN X DIRECTION FOR LINEAR INDEXING
/*!
 * @nx
 * @ny
 * @nf
 * REMARKS:
 *      - in the linear index space feature index varies first, followed by
 *      X direction index, followed by Y direction index.
 *      - remember that:
 *      k = ky * (nf*nx) + kx * nf + kf
 */
static inline int strideX(int nx, int ny, int nf)
{
   return nf;
}
//! RETURNS STRIDE IN Y DIRECTION FOR LINEAR INDEXING
/*!
 * @nx
 * @ny
 * @nf
 * REMARKS:
 *      - in the linear index space feature index varies first, followed by
 *      X direction index, followed by Y direction index.
 *      - remember that:
 *      k = ky * (nf*nx) + kx * nf + kf
 */
static inline int strideY(int nx, int ny, int nf)
{
   return nf * nx;
}
//! RETURNS STRIDE IN Y DIRECTION FOR LINEAR INDEXING
/**
 * @nx
 * @ny
 * @nf
 * REMARKS:
 *      - in the linear index space feature index varies first, followed by
 *      X direction index, followed by Y direction index.
 *      - remember that:
 *      k = ky * (nf*nx) + kx * nf + kf
 */
static inline int strideF(int nx, int ny, int nf)
{
   return 1;
}

/**
 * Returns the k direction index of the nearest neighbor in post-synaptic layer
 *
 * @kzPre the presynaptic index (can be either local or global)
 * @zScaleLog2Pre the log2 scale factor for the presynaptic layer
 * @zScaleLog2Post the log2 scale factor for the postsynaptic layer
 *    - e.g. if zScaleLog2 == 1 then dz == 2, if zScaleLog2 == -1 then dz == 1/2
 *
 */
static inline int nearby_neighbor(int kzPre, int zScaleLog2Pre, int zScaleLog2Post)
{
   int relScale = zScaleLog2Pre - zScaleLog2Post;
   float a = powf(2.0f, (float) relScale);
   return (int) (kzPre * a) + (int) (0.5f * a);
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
      int s = (int) powf(2.0, (float) -scale);
      return 0.5f * (float) (1 - s);
   }
   else {
      // post-synaptic layer has larger size scale
      int s = (int) powf(2.0, (float) scale);
      return 0.5f * (1.0f - (1.0f + 2.0f * (kPre%s)) / s);
   }
}
#endif /* DEPRECATED_FEATURES */

//! RETURNS LINEAR INDEX IN THE EXTENDED SPACE FROM INDICES IN RESTRICTED SPACE
/*!
 * @k
 * @nx
 * @ny
 * @nf
 * @nb
 * k is the index in the restricted space
 * REMARKS:
 *   - the linear indexing of neurons is done by varying first along these directions:
 *   feature direction, X direction, Y direction.
 *   - for given indices kf,kx,ky, the linear index k is given by:
 *     k = (ky-1)*(nf*nx)+(kx-1)*nf+kf
 *   - kx is the X direction index in extended space
 *   - ky is the Y direction index in extended space
 *   - kx is the X direction index in extended space
 *   .
 */
static inline int kIndexExtended(int k, int nx, int ny, int nf, int nb)
{
   int kx = nb + kxPos(k, nx, ny, nf);
   int ky = nb + kyPos(k, nx, ny, nf);
   int kf = featureIndex(k, nx, ny, nf);
   return kIndex(kx, ky, kf, nx + 2.0f*nb, ny + 2.0f*nb, nf);
}

/**
 * @kl
 * @loc
 * @nf
 */
// TODO - put back in nx,ny,... so that it will vectorize with vector of kl's
static inline int globalIndexFromLocal(int kl, PVLayerLoc loc, int nf)
{
   int kxg = loc.kx0 + kxPos(kl, loc.nx, loc.ny, nf);
   int kyg = loc.ky0 + kyPos(kl, loc.nx, loc.ny, nf);
   int  kf = featureIndex(kl, loc.nx, loc.ny, nf);
   return kIndex(kxg, kyg, kf, loc.nxGlobal, loc.nyGlobal, nf);
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
   int kx = nearbyintf((x - x0)/dx - 0.5f);
   int ky = nearbyintf((y - y0)/dy - 0.5f);
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
#endif

#endif /* CONVERSIONS_H_ */
