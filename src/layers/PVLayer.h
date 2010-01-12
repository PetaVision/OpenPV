/*
 * HyPerLayer.h
 *
 *  Created on: Jul 29, 2008
 *
 */

#ifndef HYPERLAYER_H_
#define HYPERLAYER_H_

#include "../include/pv_common.h"
#include "../utils/conversions.h"

typedef enum {
   TypeGeneric,
   TypeRetina,
   TypeSimple,
   TypeV1Simple,
   TypeV1Simple2,
   TypeV1FlankInhib,
   TypeV1FeedbackInhib,
   TypeV1Feedback2Inhib,
   TypeV1SurroundInhib
} PVLayerType;

/**
 * PVLayer is a collection of neurons of a specific class
 */
typedef struct PVLayer_ {
   int layerId;    // unique ID that identifies this layer in column
   int columnId;   // column ID
   int numNeurons; // # neurons in this HyPerLayer (i.e. in PVLayerCube)
   int numExtended;// # neurons in layer including extended border regions
   int numFeatures;// # features in this layer

   unsigned int   numActive;      // # neurons that fired
   unsigned int * activeIndices;  // indices of neurons that fired
   FILE         * activeFP;       // file of sparse activity

   // TODO - deprecate?
   PVLayerType layerType;  // the type/subtype of the layer (ie, Type_V1Simple2)

   PVLayerLoc loc;
   int   xScale, yScale;   // scale (2**scale) by which layer (dx,dy) is expanded
   float dx, dy;           // distance between neurons in the layer
   float xOrigin, yOrigin; // origin of the layer (depends on iCol)

   // Output activity buffers -- a ring buffer to implement delay
   // TODO - get rid of this, belongs in connection?
   int numDelayLevels; // # of delay levels for activity buffers

   PVLayerCube * activity;  // activity buffer FROM this layer
   float * prevActivity;    // time of previous activity

   pvdata_t * V;    // membrane potential
   pvdata_t * Vth;

   pvdata_t ** G;    // master pointer to all variable conductances (one for each phi)
   pvdata_t *  G_E;  // fast exc conductance (convenience pointer to G[PHI_EXC])
   pvdata_t *  G_I;  // fast inh conductance (convenience pointer to G[PHI_INH])
   pvdata_t *  G_IB; // slow inh (GABAB) conductance (    pointer to G[PHI_INHB])

   int numPhis; // how many membrane updates we have
   pvdata_t ** phi; // membrane update

   int numParams;
   float * params; // layer-specific parameters
   int (* updateFunc)(struct PVLayer_ * l);
   int (* initFunc)(struct PVLayer_ * l); // this is called (if it exists) just after the params are set

} PVLayer;

#ifdef __cplusplus
extern "C" {
#endif

PVLayer * pvlayer_new(int xScale, int yScale,
                      int nx, int ny, int numFeatures, int nBorder);
int pvlayer_init(PVLayer* l, int xScale, int yScale,
                 int nx, int ny, int numFeatures, int nBorder);
int pvlayer_initGlobal(PVLayer * l, int colId, int colRow, int colCol, int nRows, int nCols);
int pvlayer_initFinish(PVLayer * l);
int pvlayer_finalize(PVLayer * l);

int pvlayer_copyUpdate(PVLayer * l);

// static, hopefully fast, routines:

//static inline int pvlayer_getPos(PVLayer * l, int k, float * x, float * y, float * kf)
//{
//   *x = xPos(k, l->xOrigin, l->dx, l->loc.nx, l->loc.ny, l->numFeatures);
//   *y = yPos(k, l->yOrigin, l->dy, l->loc.nx, l->loc.ny, l->numFeatures);
//   *kf = featureIndex(k, l->loc.nx, l->loc.ny, l->numFeatures);
//
//   return 0;
//}

float pvlayer_getWeight(float x0, float x, float r, float sigma);

int pvlayer_setParams(PVLayer * l, int numParams, size_t sizeParams, void * params);
int pvlayer_getParams(PVLayer * l, int * numParams, float ** params);
int pvlayer_setFuncs (PVLayer * l, void * updateFunc, void * initFunc);

PVLayerCube * pvcube_new(PVLayerLoc * loc, int numItems);
int           pvcube_delete(PVLayerCube * cube);
int           pvcube_setAddr(PVLayerCube * cube);

PVPatch * pvpatch_new(int nx, int ny, int nf);
int       pvpatch_delete(PVPatch * p);

PVPatch * pvpatch_inplace_new(int nx, int ny, int nf);
int       pvpatch_inplace_delete(PVPatch * p);

int pvpatch_accumulate(int nk, float * v, float a, float * w);

int pvpatch_update_plasticity_incr(int nk, float * RESTRICT p,
                                   float aj, float decay, float fac);
int pvpatch_update_weights(int nk, float * RESTRICT w, float * RESTRICT m, float * RESTRICT p,
                           float aPre, float * RESTRICT aPost, float dWmax, float wMax);


int pvlayer_outputState(PVLayer * l); // default implementation: stats and activity files

#ifdef __cplusplus
}
#endif

#endif /* HYPERLAYER_H_ */
