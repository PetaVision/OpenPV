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
// #define PV_EXIT_NORMALLY 3 // Not currently used
#define PV_POSTPONE 4
#define PV_CONTINUE 5
#define PV_MARGINWIDTH_FAILURE 65

// For debugging/control:
#undef  DEBUG_OUTPUT
#undef  DEBUG_WEIGHTS
#define DEBUG 0

#define MAX_FILESYSTEMCALL_TRIES 5

// Misc:
#define PI              3.1415926535897932

// number in communicating neighborhood
#define NUM_NEIGHBORHOOD 9

// directional indices
//#define LOCAL     0 //#define NORTHWEST 1
//#define NORTH     2
//#define NORTHEAST 3
//#define WEST      4
//#define EAST      5
//#define SOUTHWEST 6
//#define SOUTH     7
//#define SOUTHEAST 8

// Limits:
#define MAX_NEIGHBORS                   8
#define INITIAL_LAYER_ARRAY_SIZE       10
#define INITIAL_CONNECTION_ARRAY_SIZE  10
#define INITIAL_PUBLISHER_ARRAY_SIZE  INITIAL_LAYER_ARRAY_SIZE
#define INITIAL_SUBSCRIBER_ARRAY_SIZE INITIAL_LAYER_ARRAY_SIZE
#define RESIZE_ARRAY_INCR               5
#define MAX_F_DELAY                    1001//21 // can have 0:MAX_F_DELAY-1 buffers of delay

#define DISPLAY_PERIOD 1.0

#endif /* PV_COMMON_H */
