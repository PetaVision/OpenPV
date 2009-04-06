/*
 * cocirc_k.h
 *
 *  Created on: Aug 13, 2008
 *      Author: dcoates
 */

// Corcircularity with curvature sensitivity

#ifndef COCIRC_K_H_
#define COCIRC_K_H_

#include "PVLayer.h"

typedef struct cocircK_params_ {
   float CCK_DEBUG;
   float CCK_usePBC;
   float CCK_NX;
   float CCK_NY;
   float CCK_DTH;

   float CCK_POST_N;
   float CCK_PRE_N;
   float CCK_POST_NO;
   float CCK_PRE_NO;
   float CCK_POST_NK;
   float CCK_PRE_NK;
   float CCK_BOUNDARY;
   float CCK_SIG_D2;
   float CCK_SIG_P2;
   float CCK_SCALE;
   float CCK_INHIB_FRACTION;
   float CCK_INHIBIT_SCALE;
   float CCK_CURVE;
   float CCK_SIG_K2;
   float CCK_SELF;
} cocircK_params;

#ifdef __cplusplus
extern "C" {
#endif

#if 0
// Pre has curvature, post does not:
int cocirc2_rcv( PVLayer* pre, PVLayer* post, int nActivity, float *activity, void *params);
// Post has curvature, pre does not:
int cocirc3_rcv( PVLayer* pre, PVLayer* post, int nActivity, float *activity, void *params);
// Neither has curvature:
int cocirc4_rcv( PVLayer* pre, PVLayer* post, int nActivity, float *activity, void *params);
#endif
// Pre & post are full curvature:
int cocircK_rcv(PVLayer* pre, PVLayer* post, float *phi, int nActivity, float *fActivity,
                void *params);
int cocircK2_rcv(PVLayer* pre, PVLayer* post, float *phi, int nActivity,
                 float *fActivity, void *params);

#ifdef __cplusplus
}
#endif

#endif /* COCIRC_H_ */
