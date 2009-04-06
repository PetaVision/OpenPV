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
/**
 * Return the feature index for the given k index
 * @k the k index (can be either global or local depending on if nx,ny are global or local)
 * @nx the number of neurons in the x direction
 * @ny the number of neurons in the y direction
 * @nf the number of neurons in the feature direction
 */
//#pragma FTT elemental, vectorize
static inline float featureIndex(int k, float nx, float ny, float nf)
{
//   return k % (int) nf;
   return fmodf((float)k, nf);
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
   return floorf( fmodf( floorf((float) k / nf), nx ) );
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
   return floorf( (k / (nx*nf)) );
}

//#pragma FTT elemental, vectorize
static inline float xPos(int k, float x0, float dx, float nx, float ny, float nf)
{
    float kx = kxPos(k, nx, ny, nf);
    return x0 + dx*(0.5f + kx);
}

//#pragma FTT elemental, vectorize
static inline float yPos(int k, float y0, float dy, float nx, float ny, float nf)
{
    float ky = kyPos(k, nx, ny, nf);
    return y0 + dy*(0.5f + ky);
}

static inline int kIndex(float kx, float ky, float kf, float nx, float ny, float nf)
{
   return (int)kf + ((int)kx + (int)ky * (int)nx) * (int)nf;
}

static inline int strideX(float nx, float ny, float nf)
{
   return (int)nf;
}

static inline int strideY(float nx, float ny, float nf)
{
   return (int)nf * (int)nx;
}

static inline int strideF(float nx, float ny, float nf)
{
   return 1;
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

//#pragma FTT elemental, vectorize
static inline float xPos(int k, float x0, float dx, float nx, float ny, float nf)
{
    float kx = kxPos(k, nx, ny, nf);
    return x0 + dx*(0.5f + kx);
}

//#pragma FTT elemental, vectorize
static inline float yPos(int k, float y0, float dy, float nx, float ny, float nf)
{
    float ky = kyPos(k, nx, ny, nf);
    return y0 + dy*(0.5f + ky);
}

// TODO - should kx,... be integer
static inline int kIndex(float kx, float ky, float kf, float nx, float ny, float nf)
{
   return kx + (ky + kf * ny) * nx;
}

static inline int strideX(float nx, float ny, float nf)
{
   return 1;
}

static inline int strideY(float nx, float ny, float nf)
{
   return (int)nx;
}

static inline int strideF(float nx, float ny, float nf)
{
   return (int)nx * (int)ny;
}

#endif // FEATURES_LAST

static inline int kIndexExtended(int k, float nx, float ny, float nf, float nb)
{
   float kx = nb + kxPos(k, nx, ny, nf);
   float ky = nb + kyPos(k, nx, ny, nf);
   float kf = featureIndex(k, nx, ny, nf);
   return kIndex(kx, ky, kf, nx + 2.0f*nb, ny + 2.0f*nb, nf);
}

// TODO - put back in nx,ny,... so that it will vectorize with vector of kl's
static inline int globalIndexFromLocal(int kl, PVLayerLoc loc, float nf)
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
