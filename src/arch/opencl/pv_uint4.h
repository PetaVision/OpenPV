/*
 * pv_uint.h
 *
 *  Created on: Jan 2, 2011
 *      Author: Craig Rasmussen
 */

#ifndef PV_UINT4_H_
#define PV_UINT4_H_

#include <limits.h>

#define CL_RANDOM_MAX       UINT_MAX

typedef struct
  {
    unsigned int s1, s2, s3;
  }
taus_state_t;

typedef struct uint4_ {
   unsigned int s0;
   taus_state_t state;
} uint4;

#endif /* PV_UINT4_H_ */
