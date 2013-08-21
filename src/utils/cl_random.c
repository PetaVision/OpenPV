/*
 * cl_random.c
 *
 *  Created on: Dec 29, 2010
 *      Author: Craig Rasmussen
 *
 *  See copyright and license information below.
 *
 *  Modified to use unsigned ints rather than long to
 *  save space on OpenCL device.
 */

/* rng/taus.c
 *
 * Copyright (C) 1996, 1997, 1998, 1999, 2000, 2007 James Theiler, Brian Gough
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */

#include "../arch/opencl/pv_uint4.h"
#include "cl_random.h"

static inline unsigned int taus_get (taus_state_t *vstate);
static void taus_set (taus_state_t *state, unsigned int s);

uint4 cl_random_get(uint4 state)
{
   state.s0 = taus_get(&state.state);
   return state;
}

// double cl_random_prob(uint4 * state) {
//    *state = cl_random_get(*state);
//    return (double) state->s0/(((double) UINT_MAX)+1);
// }

int cl_random_init(uint4 * state, size_t count, unsigned int seed)
{
   int i;

   // Commented out Nov 29, 2012
   // uint4 * state = (uint4 *) malloc(count * sizeof(uint4));
   // assert(state != NULL);

   // a zero seed can cause problems (see taus_set)
   seed = (seed == 0) ? 1 : seed;

   // initialize state array using a separate seed for each element
   //
   for (i = 0; i < count; i++) {
      taus_set(&state[i].state, i+seed);
      state[i].s0 = (state[i].state.s1 ^ state[i].state.s2 ^ state[i].state.s3);
   }

   return 0;
}

static void
taus_set (taus_state_t * state, unsigned int s)
{
  if (s == 0) {
    s = 1;      /* default seed is 1 */
  }

// original for unsigned long int
//#define LCG(n) ((69069 * n) & 0xffffffffUL)
#define LCG(n) ((69069 * n))

  state->s1 = LCG (s);
  state->s2 = LCG (state->s1);
  state->s3 = LCG (state->s2);

  /* "warm it up" */
  taus_get (state);
  taus_get (state);
  taus_get (state);
  taus_get (state);
  taus_get (state);
  taus_get (state);
  return;
}

static inline unsigned int
taus_get (taus_state_t * state)
{

//#define MASK 0xffffffffUL
//#define TAUSWORTHE(s,a,b,c,d) (((s &c) <<d) &MASK) ^ ((((s <<a) &MASK)^s) >>b)
#define TAUSWORTHE(s,a,b,c,d) (((s &c) <<d)) ^ ((((s <<a))^s) >>b)

  state->s1 = TAUSWORTHE (state->s1, 13, 19, 4294967294, 12);
  state->s2 = TAUSWORTHE (state->s2, 2 , 25, 4294967288, 4);
  state->s3 = TAUSWORTHE (state->s3, 3 , 11, 4294967280, 17);

  return (state->s1 ^ state->s2 ^ state->s3);
}


/* boxmuller.c           Implements the Polar form of the Box-Muller
                         Transformation

                      (c) Copyright 1994, Everett F. Carter Jr.
                          Permission is granted by the author to use
                          this software for any application provided this
                          copyright notice is preserved.

*/
// Argument uint4 * rnd_state added so that cl_random_prob() could be used in place of Carter's ranf().
float cl_box_muller(float m, float s, uint4 * rnd_state)      /* normal random variate generator */
{                                       /* mean m, standard deviation s */
   float x1, x2, w, y1;
   static float y2;
   static int use_last = 0;

   if (use_last)                   /* use value from previous call */
   {
      y1 = y2;
      use_last = 0;
   }
   else
   {
      do {
         *rnd_state = cl_random_get(*rnd_state);
         x1 = 2.0 * rnd_state->s0/(double) CL_RANDOM_MAX - 1.0;
         *rnd_state = cl_random_get(*rnd_state);
         x2 = 2.0 * rnd_state->s0/(double) CL_RANDOM_MAX - 1.0;
         w = x1 * x1 + x2 * x2;
      } while ( w >= 1.0 );

      w = sqrt( (-2.0 * log( w ) ) / w );
      y1 = x1 * w;
      y2 = x2 * w;
      use_last = 1;
   }

   return( m + y1 * s );
}

