#undef DEBUG_PRINT

#ifdef DEBUG_PRINT
#  include <stdio.h>
#endif


//
// Simple compute kernel that computes a convolution over an input array 
//
__kernel void convolve (
    __global float * idata,
    __global float * odata,
    const unsigned int nxGlobal,
    const unsigned int nyGlobal,
    const unsigned int nPad,
    __local float * ldata_ex)
{
   const int lStride = get_local_size(0) + 2*nPad;
   const int gStride = nxGlobal + 2*nPad;
   //const int gStride = get_global_size(0) + 2*nPad;

   const int g_ex_offset =   ( get_group_id(0) * get_local_size(0) )
                           + ( get_group_id(1) * get_local_size(1) ) * gStride;

   const unsigned int kx = get_global_id(0);
   const unsigned int ky = get_global_id(1);
   //const unsigned int k  = kx + ky*(nxGlobal);
   const unsigned int ko  = kx + ky*(nxGlobal);
   //const unsigned int k  = kx + ky*(nxGlobal + 2*nPad);
   const unsigned int k  = kx + ky*gStride;
   //const unsigned int k  = kx + ky*lStride;
   //const unsigned int kex  = kx+nPad + (ky+nPad)*(nxGlobal+2*nPad);
   const unsigned int kex  = kx+nPad + (ky+nPad)*gStride;

   const unsigned int kxl = get_local_id(0);
   const unsigned int kyl = get_local_id(1);
   //const unsigned int kl  = kxl + kyl*(get_local_size(0) + 2*nPad);
   const unsigned int kl  = kxl + kyl*lStride;
   //const int k  = kxl + kyl*gStride + g_ex_offset;

   //const unsigned int klex = (kxl + nPad) + (kyl + nPad)*(get_local_size(0) + 2*nPad);
   const unsigned int klex = (kxl + nPad) + (kyl + nPad)*lStride;


   // copy extended region
   //
   // WARNING - this only works if 2*nPad <= NXL, 2*nPad <= NYL
   //
   if (ko < nxGlobal*nyGlobal) {
      // copy NXL portion of each row
      //ldata_ex[kxl + kyl*(get_local_size(0) + 2*nPad)]
      //             = idata[kx + ky*(nxGlobal + 2*nPad)];
      ldata_ex[kl] = idata[k];
      // copy leftover section on the right of each row
      if (kxl < 2*nPad) {
         //ldata_ex[kxl + get_local_size(0) + kyl*(get_local_size(0) + 2*nPad)]
         //          = idata[kx + get_local_size(0) + ky*(nxGlobal + 2*nPad)];
         ldata_ex[kl + get_local_size(0)] = idata[k + get_local_size(0)];
      }
      // copy leftover section on the bottom
      if (kyl < 2*nPad) {
         //ldata_ex[kxl + (kyl + get_local_size(1))*(get_local_size(0) + 2*nPad)]
         //          = idata[kx + (ky + get_local_size(1))*(nxGlobal + 2*nPad)];
         ldata_ex[kl + get_local_size(1)*lStride] = idata[k + get_local_size(1)*gStride];
         // copy leftover section on the right
         if (kxl < 2*nPad) {
            ldata_ex[kl + get_local_size(0) + get_local_size(1)*lStride]
                   = idata[k + get_local_size(0) + get_local_size(1)*gStride];
            //ldata_ex[kxl + get_local_size(0) + (kyl + get_local_size(1))*(get_local_size(0) + 2*nPad)]
            //       = idata[kx + get_local_size(0) + (ky + get_local_size(1))*(nxGlobal + 2*nPad)];
         }
      }
   }

   barrier(CLK_LOCAL_MEM_FENCE);
   
   // copy local block back to global memory
   //
   //if (k < nxGlobal*nyGlobal) {
   //   odata[k] = (          ldata_ex[klex-(get_local_size(0)+2*nPad)]
   //                + ldata_ex[klex-1] + ldata_ex[klex] + ldata_ex[klex+1]
   //                +        ldata_ex[klex+(get_local_size(0)+2*nPad)]       ) / 5.0f;
   //}
   if (ko < nxGlobal*nyGlobal) {
      float sum = 0.0f;
      const int sy = get_local_size(0) + 2*nPad;
      //const int sy = nxGlobal + 2*nPad;

      for (int i = 0; i < 3; i++) {
         int offset = (i-1)*sy;
         //sum = ldata_ex[klex];
         sum += 1.1f*ldata_ex[klex+offset-1] + 2.1f*ldata_ex[klex+offset] + 3.1f*ldata_ex[klex+offset+1];
      }

      odata[ko] = sum;
   }
}
