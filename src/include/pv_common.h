/*
 * pv_common.h
 *
 *  Created on: Aug 3, 2008
 *      Author: dcoates
 */

#ifndef PV_COMMON_H
#define PV_COMMON_H

#include "pv_arch.h"
#include <assert.h>
#include <errno.h>
#include <stddef.h>
#include <stdlib.h>

// Common include file for Petavision

// Return value for successful/unsuccessful function return
#define PV_SUCCESS 0
#define PV_FAILURE 1
#define PV_BREAK 2
#define PV_POSTPONE 4

// For debugging/control:
#undef DEBUG_OUTPUT
#undef DEBUG_WEIGHTS
#define DEBUG 0

#define MAX_FILESYSTEMCALL_TRIES 5

// Misc:
#define PI 3.1415926535897932f

// number in communicating neighborhood
#define NUM_NEIGHBORHOOD 9

// Limits:
#define INITIAL_LAYER_ARRAY_SIZE 10
#define INITIAL_CONNECTION_ARRAY_SIZE 10
#define RESIZE_ARRAY_INCR 5

#endif /* PV_COMMON_H */
