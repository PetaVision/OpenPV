#undef DEBUG_PRINT

#ifdef DEBUG_PRINT
#  include <stdio.h>
#endif


//
// Simple compute kernel that computes a convolution over an input array 
//
__kernel void convolve_cpu (
    __global float * idata,
    __global float * odata,
    const unsigned int nxGlobal,
    const unsigned int nyGlobal,
    const unsigned int nPad,
    __local float * ldata_ex)
{
   const unsigned int kx = get_global_id(0);
   const unsigned int ky = get_global_id(1);
   const unsigned int k  = kx + ky*nxGlobal;

   const unsigned int kex = (kx + nPad) + (ky + nPad)*(nxGlobal + 2*nPad);

   // perform convolution
   //
   if (k < nxGlobal*nyGlobal) {
      float sum = 0.0f;
      const int sy = nxGlobal + 2*nPad;

      for (int i = 0; i < 3; i++) {
         int offset = (i-1)*sy;
         sum += 1.1f*idata[kex+offset-1] + 2.1f*idata[kex+offset] + 3.1f*idata[kex+offset+1];
      }
      odata[k] = sum;
   }

#ifdef DEBUG_PRINT
   if (ky == 4) {
      printf("idata[%d]==%f odata[%d]==%f k==%d kx==%d ky==%d\n",
             k, idata[k], k, odata[k], k, get_global_id(0), get_global_id(1));
   }

   if (k == 0) {
      printf("    local_idx==%d local_idy==%d\n", get_local_id(0), get_local_id(1));
      printf("    global_idx==%d global_idy==%d\n", get_global_id(0), get_global_id(1));
	  float4 o4 = f4[0] + f4[0];
	 // printf("odata[%d]==%f %f %f %f\n", k, o4.x, o4.y, o4.z, o4.w);
	  printf("ldata[%d]==%f idata==%f\n", k, ldata_ex[0], idata[k]);
   }
#endif

}
