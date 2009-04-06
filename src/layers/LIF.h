/*
 * LIF.h
 *
 *  Created on: Aug 5, 2008
 *      Author: dcoates
 */

#ifndef LIF_H_
#define LIF_H_

#include "PVLayer.h"

typedef struct LIF_params_ {
   float LIF_V_TH_0;
   float LIF_NOISE_FREQ;
   float LIF_NOISE_AMP;
   float LIF_SELF_EXCITATION_ATTENUATION_SCALE;
   float LIF_DT_d_TAU;
   float LIF_V_Min;

} LIF_params;

#ifdef __cplusplus
extern "C" {
#endif

int LIF_update(PVLayer *l);
int LIF2_update_finish(PVLayer *l);

#ifdef __cplusplus
}
#endif

#endif /* LIF_H_ */
