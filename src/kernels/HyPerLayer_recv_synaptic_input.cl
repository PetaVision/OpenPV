#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

#include <stdio.h>

#define KX  get_global_id(0)
#define KY  get_global_id(1)
#define KXL get_local_id(0)
#define KYL get_local_id(1)

#define NX  get_global_size(0)
#define NY  get_global_size(1)

/**
 * update post-synaptic conductance based on pre-synaptic activity
 */
__kernel void HyPerLayer_recv_synaptic_input (
          int nxPre,
          int nyPre,
          int nfPre,
          int nbPre,
          int nxp,
          int nyp,
          int nfp,
          float xScale,
          float yScale,
          size_t offsetA,
          __global float * A,
          __global float * W,
          __global int   * G   // must be an integer for atomic updates
       )
{
   // This kernel is to be run over the extended pre-synaptic layer
   // The easiest way to connect to post-synaptic layer is to shift to
   // non-extended coordinates and then to scale the results
   // WARNING, here we assume # post-synaptic layer >= pre-synaptic #

   // scalar quantities
   //

   const int k_ex = KX + (nxPre + 2*nbPre)*KY;  // k index in presynaptic, extended layer

   const int kPostX = (int)(xScale*KX) - (int)(xScale*nbPre); // kPostX==0 is left boundary non-extended
   const int kPostY = (int)(yScale*KY) - (int)(yScale*nbPre); // kPostY==0 is top  boundary non-extended
   
   const int numPost = nxPre*nyPre * xScale*yScale;

   // G local extension is (1-nxp/2:nxp/2, 1-nyp/2:nyp/2)  for even?
   // G local extension is (-nxp/2:nxp/2, -nyp/2:nyp/2)  for odd
   const int x0 = -nxp/2;
   const int y0 = -nyp/2;
   
   // keep within post-synaptic, non-extended boundary
   //
   if (kPostX > -1 + x0  &&  kPostX < xScale*nxPre - x0  &&
       kPostY > -1 + y0  &&  kPostY < yScale*nyPre - y0) {

      const int gStride = xScale*nxPre;
      const int a_idx = k_ex + offsetA;
      int w_idx = nxp*nyp * k_ex;

      if (A[a_idx] > 0.0) {

         // loop over weight patch updating G atomically
         for (int j = y0; j < y0+nyp; j++) {
            int g_idx = kPostX + x0 + (kPostY + j)*gStride;

            // TODO - loop over nf as well
            for (int i = x0; i < x0+nxp; i++) {
               if (g_idx > -1 && g_idx < numPost) {
                  printf("  RCI: k_ex==%d kx_ex==%d ky_ex==%d kPostX==%d kPostY==%d g_idx==%d i==%d\n", k_ex, KX, KY, kPostX, kPostY, g_idx, i);
                  //atom_add(&G[g_idx], A[a_idx]*W[w_idx]);
                  //G[g_idx] = G[g_idx] + A[a_idx]*W[w_idx];
               }
               g_idx += 1;  // Gs assumed to be contiguous over f then x
               w_idx += 1;  // weights are assumed to be contiguous over f,x,y
            }
         }  // end nyp loop

      }     // end if activity
   }
}
