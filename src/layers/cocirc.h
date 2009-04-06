/*
 * cocirc.h
 *
 *  Created on: Aug 5, 2008
 *      Author: dcoates
 */

#ifndef COCIRC_H_
#define COCIRC_H_

#include "PVLayer.h"

typedef struct cocirc_params_ {
   float CC_DEBUG;
   float CC_NX;
   float CC_NY;
   float CC_usePBC;
   float CC_R2;
   float CC_DTH;
   float CC_SIG_C_D_x2;
   float CC_SIG_C_P_x2;
   float CC_COCIRC_SCALE;
   float CC_INHIB_FRACTION;
   float CC_INHIBIT_SCALE;
   float CC_SIG_CURV;
   float CC_DISTANT_VAL;
} cocirc_params;

#ifdef __cplusplus
extern "C" {
#endif

int cocirc_rcv(PVLayer* pre, PVLayer* post, float *phi, int nActivity, float *fActivity,
               void *params);

#ifdef __cplusplus
}
#endif

#endif /* COCIRC_H_ */
