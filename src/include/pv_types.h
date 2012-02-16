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
  CHANNEL_GAP  = 3
};

typedef struct PVPatch_ {
   pvdata_t * __attribute__ ((aligned)) data;
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

// PVAxonalArbor is being deprecated.
// Use getGSynPatch(k,arbor) instead of axonalArbor(k,arbor)->data->data
// Use getWeights(k,arbor)->{nx,ny,nf} instead of axonalArbor(k,arbor)->data->{nx,ny,nf}
// Use getPostNonextStrides()->{sx,sy,sf} instead of axonalArbor(k,arbor)->data->{sx,sy,sf}
// Use getWeights(k,arbor) instead of axonalArbor(k,arbor)->weights
// Use getPlasticIncrement(k,arbor) instead of axonalArbor(k,arbor)->plasticIncr
// Use get_dWData(k,arbor) instead of axonalArbor(k,arbor)->plasticIncr->data
// Use getGSynOffset(k,arbor) instead of axonalArbor(k,arbor)->offset
// Use getDelay(arbor) instead of axonalArbor(k,arbor)->delay
#ifdef OBSOLETE // Marked obsolete Oct 19, 2011.  See above comment block for replacement get-methods.
typedef struct PVAxonalArbor_ {
   PVPatch * data;        // data for task to work on (e.g., GSyn data)
   PVPatch * weights;     // weights to apply to the data
   PVPatch * plasticIncr; // STDP P variable
   size_t    offset;      // offset for post-synaptic activity and pDecr (STDP M variable)
   int delay; // current output delay in the associated f ring buffer (should equal fixed delay + variable delay for valid connection)
} PVAxonalArbor;

typedef struct PVAxonalArborList_ {
   unsigned int     numArbors;
   PVAxonalArbor ** arbors;
} PVAxonalArborList;
#endif

typedef struct PVPatchStrides_ {
   int sx, sy, sf;    // stride in x,y,features
} PVPatchStrides;

enum PVPatchStrideFields {
   PATCH_SX,
   PATCH_SY,
   PATCH_SF
};

/*
 * function declarations and inline definitions
 */

static inline
void pvpatch_init(PVPatch * p, int nx, int ny, int nf,
                  int sx, int sy, int sf, pvdata_t * data)
{
   p->nx = nx;
   p->ny = ny;
//   p->nf = nf;
//   p->sx = sx;
//   p->sy = sy;
//   p->sf = sf;
   p->data = data;
}

static inline
void pvpatch_adjust(PVPatch * p, int sx, int sy, int nxNew, int nyNew, int dx, int dy)
{
   p->nx = nxNew;
   p->ny = nyNew;
   p->data += dx * sx + dy * sy;
}

#endif /* PV_TYPES_H_ */
