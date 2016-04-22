#ifndef CL_RANDOM_HCL_
#define CL_RANDOM_HCL_

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

/*
 * The following adapted from rng/taus.c by Craig Rasmussen, 12/27/2010
 * for OpenCL.  Also converted (hopefully) to use 32 bit integers.
 *
 * It uses a taus_uint4 to store state and a result:
 *    state.s0 => random number
 *    state.s1 => state s1 value
 *    state.s2 => state s2 value
 *    state.s3 => state s3 value
 */

#define cl_random_get(s) (cl_taus_get(s))

static inline taus_uint4
cl_taus_get(taus_uint4 state)
{
   taus_uint4 result;

#define TAUSWORTHE(s,a,b,c,d) (((s &c) <<d)) ^ ((((s <<a))^s) >>b)

  result.state.s1 = TAUSWORTHE (state.state.s1, 13, 19, 4294967294UL, 12);
  result.state.s2 = TAUSWORTHE (state.state.s2,  2, 25, 4294967288UL, 4);
  result.state.s3 = TAUSWORTHE (state.state.s3,  3, 11, 4294967280UL, 17);

  result.s0 = (result.state.s1 ^ result.state.s2 ^ result.state.s3);

  return result;
}

static inline float cl_random_prob(taus_uint4 state)
{
   // changed (double) cast to (float) cast
   // hopefully this works, it is (partially) tested in tests/test_cl_random
   return (float) ((float) state.s0 / (float) 4294967296.0);
}

#endif // CL_RANDOM_HCL_