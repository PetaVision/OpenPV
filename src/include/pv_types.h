/*
 * pv_types.h
 *
 *  Created: Sep 10, 2008
 *  Author: dcoates
 */

#ifndef PV_TYPES_H_
#define PV_TYPES_H_

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

/**
 * PVLayerLoc describes a data items location in the global layer
 */
typedef struct PVLayerLoc_ {
   float nx, ny;
   float nxGlobal, nyGlobal; // total number of neurons in (x,y) across all hypercolumns
   float kx0, ky0;  // origin of the layer in index space
   float dx, dy;    // maybe not needed but can use for padding anyway
} PVLayerLoc;

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
 * PVLayerCube is a 3D cube (features,y,x) of a layer's data,
 *    plus location information
 */
typedef struct PVLayerCube_ {
   size_t     size;      // sizeof entire cube in bytes
   int        numItems;  // number of items in data buffer
   pvdata_t * data;      // pointer to data (may follow header)
   int        padding[NUM_PADDING];   // header size should be n*128 bits
   PVLayerLoc loc;       // location of cube in global layer
} PVLayerCube;

typedef struct PVSynapseTask_ {
   PVPatch * data;        // data for task to work on (e.g., phi data)
   PVPatch * weights;     // weights to apply to the data
   PVPatch * plasticIncr; // patch for STDP
   float   * activity;    // post-synaptic activity
} PVSynapseTask;

typedef struct PVSynapseBundle_ {
   unsigned int numTasks;
   PVSynapseTask ** tasks;
} PVSynapseBundle;

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
