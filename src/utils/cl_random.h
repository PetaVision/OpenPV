/*
 * cl_random.h
 *
 *  Created on: Dec 30, 2010
 *      Author: Craig Rasmussen
 */

#ifndef CL_RANDOM_H_
#define CL_RANDOM_H_

#include "../arch/opencl/pv_uint4.h"
#include <stdlib.h>
#include <limits.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct
  {
    unsigned int s1, s2, s3;
  }
taus_state_t;

int cl_random_init(uint4 * state, size_t count, unsigned int seed);
uint4 cl_random_get(uint4 state);
static inline double cl_random_prob(uint4 * state){*state = cl_random_get(*state);return (double) state->s0/(((double) UINT_MAX)+1);} // Why can't the statements be in cl_random.c?

#ifdef __cplusplus
}
#endif

#endif /* CL_RANDOM_H_ */
