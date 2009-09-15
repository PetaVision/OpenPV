/*
 * pv_common.h
 *
 *  Created on: Aug 3, 2008
 *      Author: dcoates
 */

#ifndef PV_COMMON_H
#  define PV_COMMON_H

#include "pv_arch.h"

// Common "C" includes for new HyPerStructure Petavision

// For debugging/control:
#undef  DEBUG_OUTPUT
#undef  DEBUG_WEIGHTS
#define LIF_STATS 1
#define DEBUG 0

// time value for saving final output
#define FINAL_TIME (-1)

// pad layer activity with border
#undef EXTEND_BORDER_INDEX

// TODO: move to HyPerCol, set as runtime param, link each layer back to its HyPerCol
// numerical integration parameters
#define DELTA_T 1.0 //time step size (msec)
#define EXPLICIT_EULER 0
#define IMPLICIT_EULER 1
#define EXACT_LINEAR 2
#define INTEGRATION_METHOD EXACT_LINEAR

// TODO: move these into the layers
#define NX 64 // 128
#define NY 64 // 128
#define NO 8
#define NK 4
#define DTH (180.0/NO)
#define DK (1.0/(6*(NK-1)))   /*1/(sqrt(DX*DX+DY*DY)*(NK-1))*/         //  change in curvature

// Misc:
#define eventtype_t float
#define RAD_TO_DEG_x2   (2.0*180.0/PI)
#define RAD_TO_DEG      (180. / PI)
#define DEG_TO_RAD      (PI/180.0)
#define PI              3.1415926535897932

//channels
#define NUM_CHANNELS 3
#define PHI0    0       // which phi buffer to use
#define PHI1    1       // which phi buffer to use
#define PHI_EXC 0       // which phi buffer to use
#define PHI_INH 1       // which phi buffer to use
#define PHI_INHB 2

#define RMIN 0.0
#define RMAX (NX*1.414/4)
#define R2MAX (RMAX*RMAX)
#define VEL (2.0*RMAX)

// Limits:
#define MAX_NEIGHBORS    8
#define MAX_LAYERS      10
#define MAX_CONNECTIONS 20
#define MAX_F_DELAY     21 // can have 0:MAX_F_DELAY-1 buffers of delay

// TODO: As soon as the interfaces stabilize, use the type-checked/safer prototypes
//#define UPDATE_FN int (*)( int numNeurons, float *V, float *phi, float *f, void *params)
#define UPDATE_FN  void*
//#define RECV_FN int (*)( PVLayer* pre, PVLayer* post, int nActivity, float *fActivity, void *params);
#define RECV_FN  void*
//#define INIT_FN int (*)( PVConnection *);
#define INIT_FN  void*

// For IO:
#define pv_log fprintf

// This probably depends on the gnu preprocessor
#ifdef DEBUG_OUTPUT
#  define pv_debug_info(format, args...)  \
      fprintf (stdout, format , ## args); \
      fprintf (stdout, "\n"); \
      fflush  (stdout)
#else
#  define pv_debug_info(format, args...)
#endif

#ifdef ECLIPSE
#  define INPUT_PATH  "src/input/"
#  define OUTPUT_PATH "src/output/"
#else
#  define INPUT_PATH  "input/"
#  define OUTPUT_PATH "output/"
#endif
#define PARAMS_FILE "params.txt"

#endif /* PV_COMMON_H */
