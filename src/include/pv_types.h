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

// PVDatatType enum was removed Mar 29, 2018.
// Used only for the HyPerLayer dataType param, which has been removed.

typedef struct PVPatchStrides_ {
   int sx, sy, sf; // stride in x,y,features
} PVPatchStrides;

enum PVPatchStrideFields { PATCH_SX, PATCH_SY, PATCH_SF };

// PV_Stream eliminated Apr 17, 2019, in favor of FileStream.

/**
 * PVLayerCube is a 3D cube (features,x,y) of a layer's data,
 *    plus location information
 */
typedef struct PVLayerCube_ {
   // number of items in data buffer
   int numItems;

   // pointer to data (may follow header)
   float const *data;

   // location of cube in global layer
   PVLayerLoc loc;
   int isSparse;
   long const *numActive;
   void const *activeIndices;
} PVLayerCube;

typedef struct { unsigned int s1, s2, s3; } taus_state_t;

typedef struct taus_uint4_ {
   unsigned int s0;
   taus_state_t state;
} taus_uint4;

#endif /* PV_TYPES_H_ */
