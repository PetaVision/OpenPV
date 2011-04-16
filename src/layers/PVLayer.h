/*
 * PVLayer.h
 *
 *  Created on: Jul 29, 2008
 *
 */

#ifndef PVLAYER_H_
#define PVLAYER_H_

#include "../include/pv_common.h"
#include "../utils/conversions.h"
#include "../include/pv_types.h"

typedef enum {
   TypeGeneric,
   TypeImage,
   TypeRetina,
   TypeSimple,
   TypeLIFSimple,
   TypeLIFSimple2,
   TypeLIFFlankInhib,
   TypeLIFFeedbackInhib,
   TypeLIFFeedback2Inhib,
   TypeLIFSurroundInhib,
   TypeLIFGeisler,
   TypeNonspiking
} PVLayerType;

/**
 * PVLayer is a collection of neurons of a specific class
 */
typedef struct PVLayer_ {
   int layerId;    // unique ID that identifies this layer in column
   int columnId;   // column ID
   int numNeurons; // # neurons in this HyPerLayer (i.e. in PVLayerCube)
   int numExtended;// # neurons in layer including extended border regions

   unsigned int   numActive;      // # neurons that fired
   unsigned int * activeIndices;  // indices of neurons that fired
   FILE         * activeFP;       // file of sparse activity

   // TODO - deprecate?
   PVLayerType layerType;  // the type/subtype of the layer (ie, Type_LIFSimple2)

   PVLayerLoc loc;
   int   xScale, yScale;   // scale (2**scale) by which layer (dx,dy) is expanded
   float dx, dy;           // distance between neurons in the layer
   float xOrigin, yOrigin; // origin of the layer (depends on iCol)

   // Output activity buffers -- a ring buffer to implement delay
   // TODO - get rid of this, belongs in connection?
   int numDelayLevels; // # of delay levels for activity buffers

   PVLayerCube * activity;  // activity buffer FROM this layer
   float * prevActivity;    // time of previous activity

   pvdata_t * V;            // membrane potential

   void * params; // layer-specific parameters

} PVLayer;

#ifdef __cplusplus
extern "C" {
#endif

PVLayer * pvlayer_new(PVLayerLoc loc, int xScale, int yScale, int numChannels);
int pvlayer_init(PVLayer* l, PVLayerLoc loc, int xScale, int yScale, int numChannels);
int pvlayer_initGlobal(PVLayer * l, int colId, int colRow, int colCol, int nRows, int nCols);
int pvlayer_initFinish(PVLayer * l);
int pvlayer_finalize(PVLayer * l);

int pvlayer_copyUpdate(PVLayer * l);

float pvlayer_getWeight(float x0, float x, float r, float sigma);

PVLayerCube * pvcube_new(PVLayerLoc * loc, int numItems);
PVLayerCube * pvcube_init(PVLayerCube * cube, PVLayerLoc * loc, int numItems);
int           pvcube_delete(PVLayerCube * cube);
size_t        pvcube_size(int numItems);
int           pvcube_setAddr(PVLayerCube * cube);

PVPatch * pvpatch_new(int nx, int ny, int nf);
int       pvpatch_delete(PVPatch * p);

PVPatch * pvpatch_inplace_new(int nx, int ny, int nf);
int       pvpatch_inplace_delete(PVPatch * p);

int pvpatch_accumulate(int nk, float * v, float a, float * w);

int pvpatch_update_plasticity_incr(int nk, float * RESTRICT p,
                                   float aj, float decay, float fac);
int pvpatch_update_weights(int nk, float * RESTRICT w, const float * RESTRICT m,
                           const float * RESTRICT p, float aPre,
                           const float * RESTRICT aPost, float dWmax, float wMin, float wMax);
int pvpatch_update_weights_localWMax(int nk, float * RESTRICT w, const float * RESTRICT m,
                           const float * RESTRICT p, float aPre,
                           const float * RESTRICT aPost, float dWMax, float wMin, float * RESTRICT Wmax);

#ifdef __cplusplus
}
#endif

#endif /* PVLAYER_H_ */
