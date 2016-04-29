/*
 * CPTest_updateStateFunctions.h
 *
 * Static inline methods to be called by CPTestInputLayer updateState methods
 *
 *  Created on: Apr 25, 2012
 *      Author: pschultz
 */

#ifndef CPTEST_UPDATESTATEFUNCTIONS_H_
#define CPTEST_UPDATESTATEFUNCTIONS_H_

#ifndef PV_USE_OPENCL
#include <layers/updateStateFunctions.h>
#else
#define pvdata_t float
#define max_pvdata_t FLT_MAX
#define PV_SUCCESS 0
#endif // PV_USE_OPENCL


#ifndef PV_USE_OPENCL
#  define CL_KERNEL
#  define CL_MEM_GLOBAL
#  define CL_MEM_CONST
#  define CL_MEM_LOCAL
#else  /* compiling with OpenCL */
#  define CL_KERNEL       __kernel
#  define CL_MEM_GLOBAL   __global
#  define CL_MEM_CONST    __constant
#  define CL_MEM_LOCAL    __local
#  include "conversions.hcl"
#  define CHANNEL_EXC   0
#  define CHANNEL_INH   1
#  define CHANNEL_INHB  2
#  define CHANNEL_GAP   3
#endif // PV_USE_OPENCL

#endif // CPTEST_UPDATESTATEFUNCTIONS_H_

// Prototypes
static inline int updateV_CPTestInputLayer(int nbatch, int numNeurons, CL_MEM_GLOBAL pvdata_t * V);

static inline int updateV_CPTestInputLayer(int nbatch, int numNeurons, CL_MEM_GLOBAL pvdata_t * V) {
   int k;
#ifndef PV_USE_OPENCL
   for( k=0; k<numNeurons * nbatch; k++ )
#else
      k = get_global_id(0);
#endif // PV_USE_OPENCL
   {
      V[k] += 1;
   }
   return PV_SUCCESS;
}

