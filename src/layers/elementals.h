/*
 * elementals.h
 *
 *  Created on: Oct 6, 2008
 *      Author: rasmussn
 */

#ifndef ELEMENTALS_H_
#define ELEMENTALS_H_

/* define this variable if code is to be run (or transformed to be run) on a Cell processor */
//#define CELL_BE

#include "../include/pv_types.h"

#include <math.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C"
{
#endif

/*
 * The following functions are simple, static inline functions.  They have been given the
 * compiler directive elemental (with same semantics as in Fortran).  The elemental functions are
 * declared in terms of scalar arguments, but can take and return arrays.  Elemental functions
 * are vectorizable.
 */

#ifndef FEATURES_LAST
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
//#pragma FTT elemental, vectorize
static inline float featureIndex(int k, float nx, float ny, float nf)
{
//   return k % (int) nf;
   return fmodf((float)k, nf);
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
 *    a = floor(k/nf) = ky*nx + kx, and then that
 *    kx = mod(a,nx), i.e. kx is the reminder of the division of a by nx,
 *    since kx <= nx-1.
 *
 *    .
 */
//#pragma FTT elemental, vectorize
static inline float kxPos(int k, float nx, float ny, float nf)
{
   return floorf( fmodf( floorf((float) k / nf), nx ) );
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
static inline float kyPos(int k, float nx, float ny, float nf)
{
   return floorf( (k / (nx*nf)) );
}
//! RETURNS X POSITION IN PHYSICAL SPACE
/*!
 * @k
 * @x0
 * @dx
 * @nx
 * @ny
 * @nf
 */
//#pragma FTT elemental, vectorize
static inline float xPos(int k, float x0, float dx, float nx, float ny, float nf)
{
    float kx = kxPos(k, nx, ny, nf);
    return x0 + dx*(0.5f + kx);
}

/**
 * @k
 * @y0
 * @dy
 * @nx
 * @ny
 * @nf
 */
//#pragma FTT elemental, vectorize
static inline float yPos(int k, float y0, float dy, float nx, float ny, float nf)
{
    float ky = kyPos(k, nx, ny, nf);
    return y0 + dy*(0.5f + ky);
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
static inline int kIndex(float kx, float ky, float kf, float nx, float ny, float nf)
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
static inline int strideX(float nx, float ny, float nf)
{
   return (int)nf;
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
static inline int strideY(float nx, float ny, float nf)
{
   return (int)nf * (int)nx;
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
static inline int strideF(float nx, float ny, float nf)
{
   return 1;
}

/**
 * Returns the k index of the nearest neighbor in another layer
 *
 * @kPre
 * @scale
 */
static inline int nearby_neighbor(int kPre, int scale)
{
   float a = powf(2.0, (float) -scale);
   return (int) (kPre * a) + (int) (0.5f * a);
}

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
      return 0.0;
   }
   else if (scale < 0) {
      // post-synaptic layer has smaller size scale
      int s = (int) powf(2.0, (float) -scale);
      return 0.5 * (1.0 - s);
   }
   else {
      // post-synaptic layer has larger size scale
      int s = (int) powf(2.0, (float) scale);
      return 0.5 * (1.0 - (1.0 + 2.0 * (kPre%s)) / s);
   }
   return 0.0;
}

#endif // ifndef FEATURES_LAST

#ifdef FEATURES_LAST
/**
 * Return the feature index for the given k index
 * @k the k index (can be either global or local depending on if nx,ny are global or local)
 * @nx the number of neurons in the x direction
 * @ny the number of neurons in the y direction
 * @nf the number of neurons in the feature direction
 */
//#pragma FTT elemental, vectorize
static inline int featureIndex(int k, float nx, float ny, float nf)
{
   return (int) (k / (nx*ny));
}

/**
 * Return the position kx for the given k index
 * @k the k index (can be either global or local depending on if nx,ny are global or local)
 * @nx the number of neurons in the x direction
 * @ny the number of neurons in the y direction
 * @nf the number of neurons in the feature direction
 */
//#pragma FTT elemental, vectorize
static inline float kxPos(int k, float nx, float ny, float nf)
{
   return k % (int) nx;
}

/**
 * Return the position ky for the given k index
 * @k the k index (can be either global or local depending on if nx,ny are global or local)
 * @nx the number of neurons in the x direction
 * @ny the number of neurons in the y direction
 * @nf the number of neurons in the feature direction
 */
//#pragma FTT elemental, vectorize
static inline float kyPos(int k, float nx, float ny, float nf)
{
   return (int) fmod( k/nx, ny );
}

/**
 * @k
 * @x0
 * @dx
 * @nx
 * @ny
 * @nf
 */
//#pragma FTT elemental, vectorize
static inline float xPos(int k, float x0, float dx, float nx, float ny, float nf)
{
    float kx = kxPos(k, nx, ny, nf);
    return x0 + dx*(0.5f + kx);
}

/**
 * @k
 * @y0
 * @dy
 * @nx
 * @ny
 * @nf
 */
//#pragma FTT elemental, vectorize
static inline float yPos(int k, float y0, float dy, float nx, float ny, float nf)
{
    float ky = kyPos(k, nx, ny, nf);
    return y0 + dy*(0.5f + ky);
}

/**
 * @kx
 * @ky
 * @kf
 * @nx
 * @ny
 * @nf
 */
// TODO - should kx,... be integer
static inline int kIndex(float kx, float ky, float kf, float nx, float ny, float nf)
{
   return kx + (ky + kf * ny) * nx;
}

/**
 * @nx
 * @ny
 * @nf
 */
static inline int strideX(float nx, float ny, float nf)
{
   return 1;
}

/**
 * @nx
 * @ny
 * @nf
 */
static inline int strideY(float nx, float ny, float nf)
{
   return (int)nx;
}

/**
 * @nx
 * @ny
 * @nf
 */
static inline int strideF(float nx, float ny, float nf)
{
   return (int)nx * (int)ny;
}

#endif // FEATURES_LAST

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
static inline int kIndexExtended(int k, float nx, float ny, float nf, float nb)
{
   float kx = nb + kxPos(k, nx, ny, nf);
   float ky = nb + kyPos(k, nx, ny, nf);
   float kf = featureIndex(k, nx, ny, nf);
   return kIndex(kx, ky, kf, nx + 2.0f*nb, ny + 2.0f*nb, nf);
}

/**
 * @kl
 * @loc
 * @nf
 */
// TODO - put back in nx,ny,... so that it will vectorize with vector of kl's
static inline int globalIndexFromLocal(int kl, LayerLoc loc, float nf)
{
   float kxg = loc.kx0 + kxPos(kl, loc.nx, loc.ny, nf);
   float kyg = loc.ky0 + kyPos(kl, loc.nx, loc.ny, nf);
   float  kf = featureIndex(kl, loc.nx, loc.ny, nf);
   return kIndex(kxg, kyg, kf, loc.nxGlobal, loc.nyGlobal, nf);
}

/**
 * Returns position (x or y) for the given one-dimensional position index.  Layer index
 * and origin are global (not based on hyper column).
 *
 * @k the one-dimensional position index (kx or ky)
 * @origin the one-dimensional origin (x0 or y0)
 * @scale the one-dimensional scale (2**scale) by which layer is expanded (dx or dy)
 */
//#pragma FTT elemental, vectorize
static inline float pos(int k, float origin, int scale)
{
    return origin + ((float) k)*pow(2,scale);
}

/**
 * @x
 */
//#pragma FTT elemental, vectorize
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
    float abs_dx = fabs(dx);

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
static inline int globalIndex(float kf, float x, float y, float x0, float y0,
		              float dx, float dy, int nx, int ny, int nf)
{
   float kx = nearbyintf((x - x0)/dx - 0.5f);
   float ky = nearbyintf((y - y0)/dy - 0.5f);
   return kIndex(kx, ky, kf, nx, ny, nf);
}


/**
 * Return global index using floating point arithmetic
 * WARNING - this breaks if nx*ny*numFeatures > 16777217
 * @kf
 * @x
 * @y
 * @x0
 * @y0
 * @dx
 * @dy
 * @nx
 * @nf
 */
//#pragma FTT elemental, vectorize
// TODO - remove?
static inline int globalIndexf_obsolete(int kf, float x, float y, float x0, float y0,
		               float dx, float dy, int nx, int nf)
{
   float kx = nearbyintf((x - x0)/dx - 0.5f);
   float ky = nearbyintf((y - y0)/dy - 0.5f);

    // See if out of bounds of this patch
//    if (ky < 0 || ky >= l->ny) {
//        printf("WARNING globalIndexf: ky is out of bounds\n");
//    	return -1;
//    }
   return (int) nearbyintf((kx + nx*ky)*nf + kf);
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
   return expf(-0.5 * dx * dx / (sigma * sigma));
}

#ifdef __cplusplus
}
#endif

#endif /* ELEMENTALS_H_ */
