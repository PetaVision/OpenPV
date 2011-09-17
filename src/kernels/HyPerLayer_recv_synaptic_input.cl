#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

#define USE_LOCAL_MEM 1

#if USE_LOCAL_MEM < 1
//#include <stdio.h>
#endif

#define NPAD 3

#define KX  get_global_id(0)
#define KY  get_global_id(1)
#define KXL get_local_id(0)
#define KYL get_local_id(1)

#define NX  get_global_size(0)
#define NY  get_global_size(1)

/**
 * update post-synaptic conductance based on pre-synaptic activity
 */
__kernel void layer_update (
          __global int * G )
{
   const int k   = KX + KY*NX;
   const int kex = (KX + NPAD) + (KY + NPAD)*(NX + 2*NPAD);

   const int count = (NX+2*NPAD)*(NY+2*NPAD);

   if (2*k < count) {
      G[2*k]   = 0;
      G[2*k+1] = 0;
   }
}


/**
 * update post-synaptic conductance based on pre-synaptic activity
 */
__kernel void recv_synaptic_input (
          int nxp,
          int nyp,
          __global float * A,
          __global int   * G,
          __global float * W  )
{
   // scalar quantities
   const int k   = KX + KY*NX;
   const int kex = (KX + NPAD) + (KY + NPAD)*(NX + 2*NPAD);
   
   const int gStride = get_global_size(0) + 2*NPAD;
   int w_idx = nxp*nyp * (KXL + KYL*NX);
   
   if (A[k] > 0.0) {
      for (int j = 0; j < nyp; j++) {

         int g_idx = kex - NPAD + (j - NPAD)*gStride;
         
         for (int i = 0; i < nxp; i++) {
            atom_add(&G[g_idx++], A[k]*W[w_idx++]);
         }
         
      }  // end nyp loop
   }     // end if activity

}
