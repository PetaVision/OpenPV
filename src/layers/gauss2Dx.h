/*
 * gauss2Dx.h
 *
 *  Created on: Aug 5, 2008
 *      Author: dcoates
 */

#ifndef GAUSS2DX_H_
#define GAUSS2DX_H_

#include "PVLayer.h"
#include "../connections/PVConnection.h"

typedef struct gauss2Dx_params_ {
   float G_usePBC;
   float G_R2;
   float G_SIGMA; //3.0
   float G_ASPECT;
   float G_OFFSET; //in pixels
   float G_SIGMA_THETA2;
   float G_DTH_MAX;
   float G_ASYM_FLAG;
   float G_WEIGHT_SCALE;
} gauss2Dx_params;

#define GAUSS2DX_PARAMS(p) (((gauss2Dx_params*)params)->p)

#ifdef __cplusplus
extern "C" {
#endif

int gauss2Dx_init(PVConnection* con);
int gauss2Dx_rcv(PVConnection* con, PVLayer *post, int nActivity, float *fActivity);
int gauss2Dx_calcWeight(PVLayer *pre, PVLayer *post, gauss2Dx_params *params,
      float* prePos, float* postPos, float *ww);

#ifdef __cplusplus
}
#endif

#endif /* GAUSS2DX_H_ */
