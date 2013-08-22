/*
 * pv_types.h
 *
 *  Created: Sep 10, 2008
 *  Author: dcoates
 */

#ifndef PV_TYPES_H_
#define PV_TYPES_H_

#include "PVLayerLoc.h"
#include <stdlib.h>   /* for size_t */
#include <stdio.h>    /* for FILE */
#include <float.h>


#define PV_ON  1
#define PV_OFF 0

// WARNING LAYER_CUBE_HEADER_SIZE should be n*128 bits
#define LAYER_CUBE_HEADER_SIZE (sizeof(PVLayerCube))
#ifdef PV_ARCH_64
#  define NUM_PADDING 1
#  define EXPECTED_CUBE_HEADER_SIZE 80
#else
#  define NUM_PADDING 1
#  define EXPECTED_CUBE_HEADER_SIZE 64
#endif

/* The common type for data */
#define pvdata_t float
#define max_pvdata_t FLT_MAX
#define min_pvdata_t FLT_MIN

/* The common type for integer sizes (e.g. nxp patch size) */
#define pvdim_t int

enum ChannelType {
  CHANNEL_EXC  = 0,
  CHANNEL_INH  = 1,
  CHANNEL_INHB = 2,
  CHANNEL_GAP  = 3,
  CHANNEL_NORM = 4,
  CHANNEL_INVALID  = -1
};

enum GSynAccumulateType {
   ACCUMULATE_CONVOLVE = 0,
   ACCUMULATE_STOCHASTIC = 1,
   ACCUMULATE_MAXPOOLING = 2
};

typedef struct PVPatch_ {
   // pvdata_t * __attribute__ ((aligned)) data;
   unsigned int offset;
   unsigned short nx, ny;
//   int nx, ny, nf;    // number of items in x,y,features
//   int sx, sy, sf;    // stride in x,y,features
#ifndef PV_ARCH_64
   float padding;       // structure size should be 8*4 bytes
#endif
} PVPatch __attribute__ ((aligned));

/**
 * PVLayerCube is a 3D cube (features,x,y) of a layer's data,
 *    plus location information
 */
typedef struct PVLayerCube_ {
   size_t     size;      // sizeof entire cube in bytes
   int        numItems;  // number of items in data buffer
   pvdata_t * data;      // pointer to data (may follow header)
   int        padding[NUM_PADDING];   // header size should be n*128 bits
   PVLayerLoc loc;       // location of cube in global layer
} PVLayerCube;

typedef struct PVPatchStrides_ {
   int sx, sy, sf;    // stride in x,y,features
} PVPatchStrides;

enum PVPatchStrideFields {
   PATCH_SX,
   PATCH_SY,
   PATCH_SF
};

typedef struct PV_Stream_ {
   char * name;
   char * mode;
   FILE * fp;
   long   filepos;
   long   filelength;
   int    isfile; /* True or false, tells whether stream corresponds to a file */
} PV_Stream;

/*
 * function declarations and inline definitions
 */

static inline
void pvpatch_init(PVPatch * p, int nx, int ny)
{
   p->nx = nx;
   p->ny = ny;
//   p->nf = nf;
//   p->sx = sx;
//   p->sy = sy;
//   p->sf = sf;
   p->offset = 0;
}

static inline
void pvpatch_adjust(PVPatch * p, int sx, int sy, int nxNew, int nyNew, int dx, int dy)
{
   p->nx = nxNew;
   p->ny = nyNew;
   p->offset += dx * sx + dy * sy;
   // p->data += dx * sx + dy * sy;
}

#endif /* PV_TYPES_H_ */
