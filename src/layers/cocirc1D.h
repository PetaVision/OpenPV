/*
 * cocirc1D.h
 *
 *  Created on: Aug 5, 2008
 *      Author: dcoates
 */

#ifndef COCIRC1D_H_
#define COCIRC1D_H_

#include "PVLayer.h"
#include "../connections/PVConnection.h"

typedef struct cocirc1D_params_ {
   float COCIRC_usePBC;
   float COCIRC_R2;
   float COCIRC_SIGMA_DIST2; //gaussian fall off with distance (squared)
   float COCIRC_SIGMA_KURVE2; // gaussian fall off with kurvature (squared)
   float COCIRC_SIGMA_COCIRC2; // gaussian fall off with cocircularity (squared)
   float COCIRC_SELF; // 0 -> no connect if d2 == 0
   float COCIRC_WEIGHT_SCALE;
} cocirc1D_params;

#define COCIRC_PARAMS(p) (((cocirc1D_params*)params)->p)

#ifdef __cplusplus
extern "C" {
#endif

int cocirc1D_init(PVConnection* con);
int cocirc1D_rcv(PVConnection* con, PVLayer *post, int nActivity, float *fActivity);
int cocirc1D_calcWeight(PVLayer *pre, PVLayer *post, cocirc1D_params *params,
      float* prePos, float* postPos, float *ww);

#ifdef __cplusplus
}
#endif

#endif /* COCIRC1D_H_ */
