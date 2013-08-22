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
#include <math.h>

#ifdef __cplusplus
extern "C"
{
#endif

int cl_random_init(uint4 * state, size_t count, unsigned int seed);
uint4 cl_random_get(uint4 state);
static inline double cl_random_max() {return (double) CL_RANDOM_MAX;}
float cl_box_muller(float m, float s, uint4 * rnd_state);

#ifdef __cplusplus
}
#endif

#endif /* CL_RANDOM_H_ */
