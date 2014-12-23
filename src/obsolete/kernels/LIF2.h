/*
 * LIF2.h
 *
 *  Created on: Aug 5, 2008
 *      Author: dcoates
 */

#ifndef LIF2_H_
#define LIF2_H_

#include "PVLayer.h"

typedef struct LIF2_params_ {
   float Vrest;
   float Vexc;
   float Vinh;
   float VinhB;

   float tau;
   float tauE;
   float tauI;
   float tauIB;

   float VthRest;
   float tauVth;
   float deltaVth;

   float noiseFreqE;
   float noiseAmpE;
   float noiseFreqI;
   float noiseAmpI;
   float noiseFreqIB;
   float noiseAmpIB;
} LIF2_params;

#ifdef __cplusplus
extern "C"
{
#endif

   int LIF2_init(PVLayer *l);
   int LIF2_update_explicit_euler(PVLayer *l, float dt);
   int LIF2_update_implicit_euler(PVLayer *l, float dt);
   int LIF2_update_exact_linear(PVLayer *l, float dt);

#ifdef __cplusplus
}
#endif

#endif /* LIF2_H_ */
