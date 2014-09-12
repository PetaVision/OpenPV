/*
 * pv_types.h
 *
 *  Created: Sep 10, 2008
 *  Author: dcoates
 */

#ifndef PV_TYPES_H_
#define PV_TYPES_H_

#include "pv_common.h"
#include "pv_datatypes.h"
#include "PVLayerLoc.h"
#ifndef CL_KERNEL_INCLUDE
#include <stdlib.h>   /* for size_t */
#include <stdio.h>    /* for FILE */
#include <float.h>
#endif

// PV_ON and PV_OFF are never used.  Uncomment this if they become necessary again.
//#define PV_ON  1
//#define PV_OFF 0

// WARNING LAYER_CUBE_HEADER_SIZE should be n*128 bits
#define LAYER_CUBE_HEADER_SIZE (sizeof(PVLayerCube))
#ifdef PV_ARCH_64
#  define NUM_PADDING 1
#  define EXPECTED_CUBE_HEADER_SIZE 80
#else
#  define NUM_PADDING 1
#  define EXPECTED_CUBE_HEADER_SIZE 64
#endif

enum ChannelType {
  CHANNEL_EXC  = 0,
  CHANNEL_INH  = 1,
  CHANNEL_INHB = 2,
  CHANNEL_GAP  = 3,
  CHANNEL_NORM = 4,
  CHANNEL_NOUPDATE = -1
};

enum GSynAccumulateType {
   ACCUMULATE_CONVOLVE = 0,
   ACCUMULATE_STOCHASTIC = 1,
   ACCUMULATE_MAXPOOLING = 2
};

typedef struct PVPatch_ {
   unsigned int offset;
   unsigned short nx, ny;
#ifndef PV_ARCH_64
   float padding;       // structure size should be 8*4 bytes
#endif
} PVPatch __attribute__ ((aligned));

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

typedef enum {
   TypeGeneric,
   TypeImage,
   TypeRetina,
   TypeSimple,
   TypeLIFSimple,
   TypeLIFGap,
   TypeLIFHC,
   TypeLIFSimple2,
   TypeLIFFlankInhib,
   TypeLIFFeedbackInhib,
   TypeLIFFeedback2Inhib,
   TypeLIFSurroundInhib,
   TypeLIFGeisler,
   TypeBIDS,
   TypeLCA,
   TypeNonspiking
} PVLayerType;

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

/**
 * PVLayer is a collection of neurons of a specific class
 */
typedef struct PVLayer_ {
   int numNeurons; // # neurons in this HyPerLayer (i.e. in PVLayerCube)
   int numExtended;// # neurons in layer including extended border regions

   unsigned int   numActive;      // # neurons that fired
   unsigned int * activeIndices;  // indices of neurons that fired
   PV_Stream    * activeFP;       // file of sparse activity
   PV_Stream    * posFP;          // file of sparse activity frame positions

   // TODO - deprecate?
   PVLayerType layerType;  // the type/subtype of the layer (ie, Type_LIFSimple2)

   PVLayerLoc loc;
   int   xScale, yScale;   // scale (2**scale) by which layer (dx,dy) is expanded
   float dx, dy;           // distance between neurons in the layer
   float xOrigin, yOrigin; // origin of the layer (depends on iCol)

   PVLayerCube * activity;  // activity buffer FROM this layer
   float * prevActivity;    // time of previous activity

   pvdata_t * V;            // membrane potential

   void * params; // layer-specific parameters

} PVLayer;

#endif /* PV_TYPES_H_ */
