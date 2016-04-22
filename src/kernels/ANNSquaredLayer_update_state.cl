#include "../layers/updateStateFunctions.h"

#ifndef PV_USE_OPENCL
#  define CL_KERNEL
#  define CL_MEM_GLOBAL
#  define CL_MEM_CONST
#  define CL_MEM_LOCAL
#else  /* compiling with OpenCL */
#  define CL_KERNEL       __kernel
#  define CL_MEM_GLOBAL   __global
#  define CL_MEM_CONST    __constant
#  define CL_MEM_LOCAL    __local
//#  include "conversions.hcl"
#endif


//
CL_KERNEL
void ANNSquaredLayer_update_state(
    const int nbatch,
    const int numNeurons,
    const int nx,
    const int ny,
    const int nf,
    const int lt,
    const int rt,
    const int dn,
    const int up,

    CL_MEM_GLOBAL float * V,
    CL_MEM_GLOBAL float * GSynHead,
    CL_MEM_GLOBAL float * activity)
{

   updateV_ANNSquaredLayer(nbatch, numNeurons, V, GSynHead);
   setActivity_HyPerLayer(nbatch, numNeurons, activity, V, nx, ny, nf, lt, rt, dn, up);

}
