/*
 * cl_random.h
 *
 *  Created on: Dec 30, 2010
 *      Author: Craig Rasmussen
 */

#ifndef CL_RANDOM_H_
#define CL_RANDOM_H_

#include "../arch/opencl/pv_opencl.h"
#include "../arch/opencl/pv_uint4.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C"
{
#endif

uint4 * cl_random_init(size_t count);
uint4   cl_random_state(uint4 state);

#ifdef __cplusplus
}
#endif

#endif /* CL_RANDOM_H_ */
