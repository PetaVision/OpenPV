/*
 * prob_fire.h
 *
 *  Created on: Aug 5, 2008
 *      Author: dcoates
 */

#ifndef prob_fire_H_
#define prob_fire_H_

#include "PVLayer.h"

typedef struct probFire_params_ {
   float PROB_FIRE_THRESHOLD;
   float PROB_FIRE_SCALE;
   float PROB_FIRE_AMP;
   float PROB_FIRE_NOISE_FREQ;
   float PROB_FIRE_NOISE_AMP;
} probFire_params;

#ifdef __cplusplus
extern "C" {
#endif

void probFire_update(PVLayer *l);

#ifdef __cplusplus
}
#endif

#endif /* prob_fire_H_ */
