/*
 * gabor.h
 *
 *  Created on: Aug 5, 2008
 *      Author: dcoates
 */

#ifndef GABOR_H_
#define GABOR_H_

#include "PVLayer.h"

typedef struct gabor_params_ {
   float CC_NX;
   float CC_NY;
   float G_usePBC;
   float G_R2;
   float G_SIGMA; //3.0
   float GAMMA_BASE;
   float GAMMA_MULT;
   float LAMBDA_BASE;
   float LAMBDA_MULT;
   float GABOR_WEIGHT_SCALE;
   float GABOR_WEIGHT_SCALE_EXP;
} gabor_params;

#define GABOR_PARAMS(p) (((gabor_params*)params)->p)

#ifdef __cplusplus
extern "C" {
#endif

extern int gabor_rcv(PVLayer* pre, PVLayer* post, float *phi, int nActivity,
                     float *fActivity, void *params);

extern inline int gabor_calcWeight(gabor_params *params, float* prePos, float* postPos,
                                   float *ww);

#ifdef __cplusplus
}
#endif

#endif /* GABOR_H_ */
