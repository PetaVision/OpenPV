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

#include <layers/updateStateFunctions.h>

#endif // CPTEST_UPDATESTATEFUNCTIONS_H_

// Prototypes
static inline int updateV_CPTestInputLayer(int nbatch, int numNeurons, pvdata_t * V);

static inline int updateV_CPTestInputLayer(int nbatch, int numNeurons, pvdata_t * V) {
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

