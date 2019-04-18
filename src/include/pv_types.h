/*
 * pv_types.h
 *
 *  Created: Sep 10, 2008
 *  Author: dcoates
 */

#ifndef PV_TYPES_H_
#define PV_TYPES_H_

#include "PVLayerLoc.h"
#include "pv_common.h"
#include <float.h>
#include <stdio.h>

enum ChannelType {
   CHANNEL_EXC      = 0,
   CHANNEL_INH      = 1,
   CHANNEL_INHB     = 2,
   CHANNEL_GAP      = 3,
   CHANNEL_NORM     = 4,
   CHANNEL_NOUPDATE = -1
};

enum PVDataType {
   PV_FLOAT = 0,
   PV_INT   = 1,
};

typedef struct PVPatchStrides_ {
   int sx, sy, sf; // stride in x,y,features
} PVPatchStrides;

enum PVPatchStrideFields { PATCH_SX, PATCH_SY, PATCH_SF };

typedef struct PV_Stream_ {
   char *name;
   char *mode;
   FILE *fp;
   long filepos;
   long filelength;

   // True or false, tells whether stream corresponds to a file
   int isfile;

   // True or false, if true, calls to PV_fwrite will do a readback check
   int verifyWrites;
} PV_Stream;

/**
 * PVLayerCube is a 3D cube (features,x,y) of a layer's data,
 *    plus location information
 */
typedef struct PVLayerCube_ {
   // size of entire cube in bytes
   size_t size;

   // number of items in data buffer
   int numItems;

   // pointer to data (may follow header)
   float *data;

   // location of cube in global layer
   PVLayerLoc loc;
   int isSparse;
   long const *numActive;
   void const *activeIndices;
} PVLayerCube;

/**
 * PVLayer is a collection of neurons of a specific class
 */
typedef struct PVLayer_ {
   // # neurons in this layer
   int numNeurons;
   // # neurons in layer including extended border regions
   int numExtended;

   // # neurons in this layer across all batches
   int numNeuronsAllBatches;

   // # neurons in this layer across all batches, including extended regions
   int numExtendedAllBatches;

   PVLayerLoc loc;

   // Layer size = 2^(-scale) * column size.
   // Layers with positive xScale are more dense in the x dimension
   int xScale, yScale;

   PVLayerCube *activity;

   // time of previous spike for each neuron
   float *prevActivity;

   // membrane potential
   float *V;
} PVLayer;

typedef struct { unsigned int s1, s2, s3; } taus_state_t;

typedef struct taus_uint4_ {
   unsigned int s0;
   taus_state_t state;
} taus_uint4;

#endif /* PV_TYPES_H_ */
