/*
 * pv_random.c
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

#include "pv_random.h"
#include <assert.h>

static inline unsigned int taus_get (void *vstate);
static double taus_get_float (void *vstate);
static void taus_set (void *state, unsigned int s);

typedef struct
  {
    unsigned int s1, s2, s3;
  }
taus_state_t;

uint4 * pv_random_init(size_t count)
{
   int i;

   uint4 * state = (uint4 *) malloc(count * sizeof(uint4));
   assert(state != NULL);

   // initialize state array using a separate seed for each element
   //
   for (i = 0; i < count; i++) {
      taus_set(&state[0].s1, i+1);
      state[i].s0 = (state[i].s1 ^ state[i].s2 ^ state[i].s3);
   }

   return state;
}

static void
taus_set (void * vstate, unsigned int s)
{
   taus_state_t *state = (taus_state_t *) vstate;

  if (s == 0)
    s = 1;      /* default seed is 1 */

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
taus_get (void * vstate)
{
   taus_state_t *state = (taus_state_t *) vstate;

//#define MASK 0xffffffffUL
//#define TAUSWORTHE(s,a,b,c,d) (((s &c) <<d) &MASK) ^ ((((s <<a) &MASK)^s) >>b)
#define TAUSWORTHE(s,a,b,c,d) (((s &c) <<d)) ^ ((((s <<a))^s) >>b)

  state->s1 = TAUSWORTHE (state->s1, 13, 19, 4294967294, 12);
  state->s2 = TAUSWORTHE (state->s2, 2 , 25, 4294967288, 4);
  state->s3 = TAUSWORTHE (state->s3, 3 , 11, 4294967280, 17);

  return (state->s1 ^ state->s2 ^ state->s3);
}
