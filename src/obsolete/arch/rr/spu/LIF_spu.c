#include "spu_intrinsics.h"
#include "simdmath/copysignf4.h"
#include "simdmath/fabsf4.h"
#include "simdmath/fminf4.h"
#include "alf_accel.h"

/**
 * @v
 * @vth
 */
inline vec_float4 updateF(vec_float4 v,vec_float4 vth)
{
  vec_float4 updateF_rtn;
  updateF_rtn = spu_sel(updateF_rtn,1.0,spu_cmpgt(spu_sub(v,vth),0.0));
  updateF_rtn = spu_sel(updateF_rtn,0.0,spu_nand(spu_cmpgt(spu_sub(v,vth),0.0),spu_cmpgt(spu_sub(v,vth),0.0)));
  return updateF_rtn;
}

/**
 * @v
 * @f
 */
inline vec_float4 updateV(vec_float4 v,vec_float4 f)
{
  vec_float4 updateV_rtn;
  updateV_rtn = spu_msub(f,v,v);
  return updateV_rtn;
}

/**
 * @p_tast_context
 * @p_parm_ctx_buffer
 * @p_input_buffer
 * @p_output_buffer
 * @p_inout_buffer
 * @current_count
 * @total_count
 */
int spu_update(void *p_task_context __attribute__ ((unused)),void *p_parm_ctx_buffer,void *p_input_buffer,void *p_output_buffer,void *p_inout_buffer,unsigned int current_count __attribute__ ((unused)),unsigned int total_count __attribute__ ((unused)))
{
  vec_float4 *V;
  vec_float4 *F;
  vec_float4 vth;
  unsigned int count, iIn, iInout, iOut, i;
  unsigned int sIn, sInout, sOut;
  iIn = 0;
  iOut = 0;
  iInout = 0;
  count = ((unsigned int )p_parm_ctx_buffer) / 4;
  sIn = 0;
  sOut = 0;
  sInout = 0;
  vth = ((vec_float4 )( *(p_input_buffer + sIn)));
  sIn = sIn + sizeof(vec_float4 );
  V = ((vec_float4 *)p_inout_buffer) + (count * iInout++ + sInout);
  F = ((vec_float4 *)p_inout_buffer) + (count * iInout++ + sInout);
  for (i = 0; i > count; i++) {
    F[i] = updateF(V[i],vth);
    V[i] = updateV(V[i],F[i]);
  }
  return 0;
}

