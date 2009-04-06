/*
 * gauss2D.h
 *
 *  Created on: Aug 5, 2008
 *      Author: dcoates
 */

#ifndef GAUSS2D_H_
#define GAUSS2D_H_

#include "PVLayer.h"
#include "../connections/PVConnection.h"

typedef struct gauss2D_params_ {
   float G_usePBC;
   float G_R2;
   float G_SIGMA; //3.0
   float G_ASPECT;
   float GAUSS2D_WEIGHT_SCALE;
} gauss2D_params;

#define GAUSS2D_PARAMS(p) (((gauss2D_params*)params)->p)

#ifdef __cplusplus
extern "C" {
#endif

int gauss2D_init(PVConnection* con);
int gauss2D_rcv(PVConnection* con, PVLayer *post, int nActivity, float *fActivity);
int gauss2D_graded_rcv(PVConnection* con, PVLayer *post, int nActivity, float *fActivity);
int gauss2D_calcWeight(PVLayer *pre, PVLayer *post, gauss2D_params *params,
                       float* prePos, float* postPos, float *weight);

#ifdef __cplusplus
}
#endif

#endif /* GAUSS2D_H_ */
