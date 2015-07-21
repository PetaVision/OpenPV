/*
 * elementals.h
 *
 *  Created on: Nov 16, 2008
 *      Author: rasmussn
 */

#ifndef ELEMENTALS_H_
#define ELEMENTALS_H_

#include <simdmath.h>

/*--------------------elemental functions on SPU--------------------------*/

inline vec_float4 featureIndex(vec_int4 k, vec_float4 nx, vec_float4 ny, vec_float4 nf)
{
   vec_float4 fk = spu_convtf(k,0);
   return fmodf4_fast(fk, nf);
}

inline vec_float4 kxPos(vec_int4 k, vec_float4 nx, vec_float4 ny, vec_float4 nf)
{
   vec_float4 fk = spu_convtf(k,0);
   return floorf4_fast( fmodf4_fast( floorf4_fast(divf4(fk,nf)), nx ) );
}

inline vec_float4 kyPos(vec_int4 k, vec_float4 nx, vec_float4 ny, vec_float4 nf)
{
   vec_float4 fk = spu_convtf(k,0);
   return floorf4_fast( divf4(fk, spu_mul(nx,nf)) );
}

/* can be used in local frame as long as nx*ny*nf <= 10^24 */
static inline vec_int4 kIndexLocal(vec_float4 kx, vec_float4 ky, vec_float4 kf,
                                   vec_float4 nx, vec_float4 ny, vec_float4 nf)
{
   // return kf + (kx + ky * nx) * nf;
   return spu_convts(spu_madd(spu_madd(ky,nx,kx), nf, kf), 0);
}

#endif /* ELEMENTALS_H_ */
