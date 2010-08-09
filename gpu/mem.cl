//#include <OpenCL/opencl.h>

//
// simple compute kernel
//
__kernel void bandwidth (
    __global float * V,
    const uint nx,
    const uint ny,
    const uint nPad)
{
   const uint kx = get_global_id(0);
   const uint ky = get_global_id(1);
   const uint k  = kx + nx*ky;

   const uint kxl = get_local_id(0);
   const uint kyl = get_local_id(1);
   const uint kl  = kxl + (get_local_size(0) + 2*nPad) * kyl;

   float one = 1.1;

   if (k < nx*ny) {
      V[k] = one * V[k];
   }
}
