/*
 * cl_random.h
 *
 *  Created on: Dec 30, 2010
 *      Author: Craig Rasmussen
 */

#ifndef CL_RANDOM_H_
#define CL_RANDOM_H_

#include <limits.h>
#include <math.h>
#include <stdlib.h>

#define CL_RANDOM_MAX UINT_MAX

#ifdef __cplusplus
extern "C" {
#endif

typedef struct { unsigned int s1, s2, s3; } taus_state_t;

typedef struct {
   unsigned int s0;
   taus_state_t state;
} taus_uint4;

struct box_muller_state {
   taus_uint4 *state;
   int use_last;
   float last_value;
};

int cl_random_init(taus_uint4 *state, size_t count, unsigned int seed);
taus_uint4 cl_random_get(taus_uint4 state);
static inline double cl_random_max() { return (double)CL_RANDOM_MAX; }
float cl_box_muller(float m, float s, struct box_muller_state *bm_state);
static inline float cl_random_prob(taus_uint4 state) {
   return (float)((float)state.s0 / (float)4294967296.0);
}

#ifdef __cplusplus
}
#endif

#endif /* CL_RANDOM_H_ */
