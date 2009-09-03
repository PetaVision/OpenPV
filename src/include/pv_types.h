/*
 * pv_types.h
 *
 *  Created: Sep 10, 2008
 *  Author: dcoates
 */

#ifndef PV_TYPES_H_
#define PV_TYPES_H_

#include "LayerLoc.h"
#include <stdlib.h>   /* for size_t */

#define PV_ON  1
#define PV_OFF 0

// WARNING LAYER_CUBE_HEADER_SIZE should be n*128 bits
#define NUM_PADDING 1
#define LAYER_CUBE_HEADER_SIZE (sizeof(PVLayerCube))
#ifdef PV_ARCH_64
#  define EXPECTED_CUBE_HEADER_SIZE 64
#else
#  define EXPECTED_CUBE_HEADER_SIZE 48
#endif

/* The common type for data */
#define pvdata_t float

enum ChannelType {
  CHANNEL_EXC  = 0,
  CHANNEL_INH  = 1,
  CHANNEL_INHB = 2
};

typedef struct PVRect_ {
	float x;
	float y;
	float width;
	float height;
} PVRect;

typedef struct PVPatch_ {
   pvdata_t * __attribute__ ((aligned)) data;
   float nx, ny, nf;    // number of items in x,y,features
   float sx, sy, sf;    // stride in x,y,features
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
   LayerLoc loc;       // location of cube in global layer
} PVLayerCube;

typedef struct PVAxonalArbor_ {
   PVPatch * data;        // data for task to work on (e.g., phi data)
   PVPatch * weights;     // weights to apply to the data
   PVPatch * plasticIncr; // STDP P variable
   size_t    offset;      // offset for post-synaptic activity and pDecr (STDP M variable)
} PVAxonalArbor;

typedef struct PVAxonalArborList_ {
   unsigned int     numArbors;
   PVAxonalArbor ** arbors;
} PVAxonalArborList;

/*
 * function declarations and inline definitions
 */

static inline
void pvpatch_init(PVPatch * p, int nx, int ny, int nf,
                  float sx, float sy, float sf, pvdata_t * data)
{
   p->nx = nx;
   p->ny = ny;
   p->nf = nf;
   p->sx = sx;
   p->sy = sy;
   p->sf = sf;
   p->data = data;
}

static inline
void pvpatch_adjust(PVPatch * p, int nxNew, int nyNew, int dx, int dy)
{
   p->nx = nxNew;
   p->ny = nyNew;
   p->data += dx * (int)p->sx + dy * (int)p->sy;
}

#endif /* PV_TYPES_H_ */
