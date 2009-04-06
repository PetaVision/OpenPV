/*
 * HyPerConnection.h
 *
 *  Created on: Aug 5, 2008
 *      Author: dcoates
 */

#ifndef PVCONNECTION_H_
#define PVCONNECTION_H_

#include "../layers/PVLayer.h"

/**
 * A PVConnection identifies a connection between two layers
 */
typedef struct {
   int delay; // current output delay in the associated f ring buffer (should equal fixed delay + varible delay for valid connection)
   int fixDelay; // fixed output delay. TODO: should be float
   int varDelayMin; // minimum variable conduction delay
   int varDelayMax; // maximum variable conduction delay
   int numDelay;
   int isGraded; //==1, release is stochastic with prob = (activity <= 1), default is 0 (no graded release)
   float vel;  // conduction velocity in position units (pixels) per time step--added by GTK
   float rmin; // minimum connection distance
   float rmax; // maximum connection distance
} PVConnParams;

/**
 * A PVConnection identifies a connection between two layers
 */
typedef struct PVConnection_ {
   PVLayer * pre;
   PVLayer * post;

   float * preNormF;
   float * postNormF; // normalization factor for the weights at each feature

   int delay; // current output delay in the associated f ring buffer (should equal fixed delay + varible delay for valid connection)
   int fixDelay; // fixed output delay. TODO: should be float
   float vel; // conduction velocity in position units (pixels) per time step--added by GTK
   float rmin; // minimum connection distance
   float rmax; // maximum connection distance
   int varDelayMin; // minimum variable conduction delay
   int varDelayMax; // maximum variable conduction delay
   int numDelay;
   int isGraded; //==1, release is stochastic with prob = (factivity <= 1), default is 0 (no graded release)
   int readIdx; // current read loc in the ring buffer

   int whichPhi; // which phi buffer of the post to update

   // The callback to handle messages sent on this connection:
   int (*recvFunc)(struct PVConnection_ * con, PVLayer * post, int numActivity,
         float* inActivityBuf);

   // The callback for one-time inits
   int (*initFunc)(struct PVConnection_ * con);

   // This should be populated with a struct of the appropriate
   // <connectiontype>_params type:
   void * params;

   // Memoized values for the weight cache:
   float r2; // maximum possible distance (squared) to a synapse (should be rmax*rmax)
   int numKernels; // how many kernels per pre retinotopic location
   int yStride; // used only for weights
   int xStride; // used only for weights
   int fStride; // used only for weights
   int * numPostSynapses; // number of valid synapses in contiguous lists for each pre kernel

   /* A simple synaptic weight cache organized by relative position */
   float ** weights; // array of weights organized by relative position (one array for each pre kernel)
   int ** postCacheIndices; // contiguous list of post cache indices (use to determine relative position)
   int ** postKernels; // contiguous list of post kernel indices

   // 2D look up table organized by: ( delayIdx, preKernelIdx )
   int maxNumSynapses;
   int *numSynapsesLUT;

   // 3D look up tables organized by: ( delayIdx, preKernelIdx, synapseIdx )
   float * xOffsetLUT; // x-distance to post neuron (add--with PBCs--to pre neuron location to get target xIdx)
   float * yOffsetLUT; // y-distance to post neuron (add--with PBCs--to pre neuron location to get target yIdx)
   int * postFeatureLUT; // post feature index (no translational invariance assumed in feature space)
   int * postKernelLUT; // index into postNormF
   float * weightLUT; // weight value

} PVConnection;

#ifdef __cplusplus
extern "C"
{
#endif

// "Object-oriented"-like default interface:
typedef int (*PVWeightFunction)(PVLayer * pre, PVLayer * post, void * params, float * prePos,
      float *postPos, float *weight);
int PVConnection_default_normalize(PVConnection * con, PVWeightFunction calcWeight,
      float scale);
int PVConnection_default_rcv(PVConnection * con, PVLayer * post, int nActivity,
      float *fActivity);

int pvConnInit(PVConnection * pvConn, PVLayer * pre, PVLayer * post, PVConnParams * p, int channel);
int pvConnFinalize(PVConnection * pvConn);

#ifdef __cplusplus
}
#endif

#endif /* PVCONNECTION_H_ */
