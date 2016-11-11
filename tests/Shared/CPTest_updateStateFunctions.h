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
static inline int updateV_CPTestInputLayer(int nbatch, int numNeurons, float *V);

static inline int updateV_CPTestInputLayer(int nbatch, int numNeurons, float *V) {
   int k;
   for (k = 0; k < numNeurons * nbatch; k++) {
      V[k] += 1;
   }
   return PV_SUCCESS;
}
