/*
 * thru.h
 *
 *  Created on: Aug 10, 2008
 *      Author: dcoates
 */

#ifndef THRU_H_
#define THRU_H_

#include "PVLayer.h"

typedef struct thru_params_ {
   float THRU_SCALE; // amt to scale input by before copy
   float USE_F_DIRECTLY; // whether to just do a memcpy, vs. into phi
   float DIM_EXPANSION; // stride for copy--should be at least 1
} thru_params;

#ifdef __cplusplus
extern "C" {
#endif

extern int thru_rcv(PVLayer* pre, PVLayer* post, float *phi, int nActivity,
      float *fActivity, thru_params* params);

#ifdef __cplusplus
}
#endif

#endif /* THRU_H_ */
