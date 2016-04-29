/*
 * pv_common.h
 *
 *  Created on: Aug 3, 2008
 *      Author: dcoates
 */

#ifndef PV_COMMON_H
#  define PV_COMMON_H

#include <stddef.h>
#include <stdlib.h>
#include <errno.h>
#include <assert.h>
#include "pv_arch.h"

// Common include file for Petavision

// Return value for successful/unsuccessful function return
#define PV_SUCCESS 0
#define PV_FAILURE 1
#define PV_BREAK 2
#define PV_EXIT_NORMALLY 3
#define PV_POSTPONE 4
#define PV_CONTINUE 5
#define PV_MARGINWIDTH_FAILURE 65

// For debugging/control:
#undef  DEBUG_OUTPUT
#undef  DEBUG_WEIGHTS
#define LIF_STATS 1
#define DEBUG 0

// TODO: move to HyPerCol, set as runtime param, link each layer back to its HyPerCol
// numerical integration parameters
#define DELTA_T 1.0 //time step size (msec)
#define MIRROR_BC_FLAG true

#define MAX_FILESYSTEMCALL_TRIES 5

//// control dynamics
//#define STATIC_DYNAMICS   0
//#define RATE_DYNAMICS     1
//#define FIRING_DYNAMICS   2
//#define WMAX_DYNAMICS     STATIC_DYNAMICS
//#define VTHREST_DYNAMICS  FIRING_DYNAMICS

// Misc:
#define eventtype_t float
#define RAD_TO_DEG_x2   (2.0*180.0/PI)
#define RAD_TO_DEG      (180. / PI)
#define DEG_TO_RAD      (PI/180.0)
#define PI              3.1415926535897932

//channels
#define MAX_CHANNELS 3
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
#define MAX_NEIGHBORS                   8
#define INITIAL_LAYER_ARRAY_SIZE       10
#define INITIAL_CONNECTION_ARRAY_SIZE  10
#define INITIAL_PUBLISHER_ARRAY_SIZE  INITIAL_LAYER_ARRAY_SIZE
#define INITIAL_SUBSCRIBER_ARRAY_SIZE INITIAL_LAYER_ARRAY_SIZE
#define RESIZE_ARRAY_INCR               5
#define MAX_F_DELAY                    1001//21 // can have 0:MAX_F_DELAY-1 buffers of delay

#define DISPLAY_PERIOD 1.0

// TODO: As soon as the interfaces stabilize, use the type-checked/safer prototypes
//#define UPDATE_FN int (*)( int numNeurons, float *V, float *phi, float *f, void *params)
#define UPDATE_FN  void*
//#define RECV_FN int (*)( PVLayer* pre, PVLayer* post, int nActivity, float *fActivity, void *params);
#define RECV_FN  void*
//#define INIT_FN int (*)( PVConnection *);
#define INIT_FN  void*

// This probably depends on the gnu preprocessor
#ifdef DEBUG_OUTPUT
#  define pv_debug_info(format, args...)  \
      fprintf (stdout, format , ## args); \
      fprintf (stdout, "\n"); \
      fflush  (stdout)
#else
#  define pv_debug_info(format, args...)
#endif

#define INPUT_PATH  "input/"
#define OUTPUT_PATH "output/"
#define PARAMS_FILE "params.txt"

#endif /* PV_COMMON_H */
