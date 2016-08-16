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
#include <float.h>
#include <stdio.h>

// PV_ON and PV_OFF are never used.  Uncomment this if they become necessary again.
//#define PV_ON  1
//#define PV_OFF 0

// WARNING LAYER_CUBE_HEADER_SIZE should be n*128 bits
#define LAYER_CUBE_HEADER_SIZE (sizeof(PVLayerCube))
#ifdef PV_ARCH_64
#  define NUM_PADDING 1
#  define EXPECTED_CUBE_HEADER_SIZE 104 
#else
#  define NUM_PADDING 1
#  define EXPECTED_CUBE_HEADER_SIZE 72 //Check this with new cube fields
#endif

enum ChannelType {
  CHANNEL_EXC  = 0,
  CHANNEL_INH  = 1,
  CHANNEL_INHB = 2,
  CHANNEL_GAP  = 3,
  CHANNEL_NORM = 4,
  CHANNEL_NOUPDATE = -1
};

#ifdef OBSOLETE // Marked obsolete May 3, 2016.  HyPerConn defines HyPerConnAccumulateType and PoolingConn defines PoolingType
enum GSynAccumulateType {
   ACCUMULATE_CONVOLVE = 0,
   ACCUMULATE_STOCHASTIC = 1,
   ACCUMULATE_MAXPOOLING = 2,
   ACCUMULATE_SUMPOOLING = 3,
   ACCUMULATE_AVGPOOLING = 4
};
#endif // OBSOLETE // Marked obsolete May 3, 2016.  HyPerConn defines HyPerConnAccumulateType and PoolingConn defines PoolingType

enum PVDataType{
   PV_FLOAT = 0,
   PV_INT = 1,
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
   int    verifyWrites; /* True or false, if true, calls to PV_fwrite will do a readback check.  */
} PV_Stream;

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
   int isSparse;
   long * numActive;
   unsigned int * activeIndices;
} PVLayerCube;

/**
 * PVLayer is a collection of neurons of a specific class
 */
typedef struct PVLayer_ {
   int numNeurons; // # neurons in this HyPerLayer (i.e. in PVLayerCube)
   int numExtended;// # neurons in layer including extended border regions
   int numNeuronsAllBatches; // # Total neurons in this HyPerLayer, including batches
   int numExtendedAllBatches;// # Total neurons in layer including extended border regions and batches

   //unsigned int   numActive;      // # neurons that fired
   //unsigned int * activeIndices;  // indices of neurons that fired
   PV_Stream    * activeFP;       // file of sparse activity

   PVLayerLoc loc;
   int   xScale, yScale;   // layersize=2**(-scale)*columnsize.  Layers with positive xScale are more dense in the x-dimension.

   PVLayerCube * activity;  // activity buffer FROM this layer
   float * prevActivity;    // time of previous activity

   pvdata_t * V;            // membrane potential

   void * params; // layer-specific parameters

} PVLayer;

typedef struct
  {
    unsigned int s1, s2, s3;
  }
taus_state_t;

typedef struct taus_uint4_ {
   unsigned int s0;
   taus_state_t state;
} taus_uint4;

#endif /* PV_TYPES_H_ */
