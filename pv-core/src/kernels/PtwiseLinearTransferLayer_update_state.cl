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
// update the state of an ANN layer
//
// To allow porting to GPUs, functions called from this function must be
// static inline functions.  If a subclass needs new behavior, it needs to
// have its own static inline function.
//
CL_KERNEL
void PtwiseLinearTransferLayer_update_state(
    const int numNeurons,
    const int nx,
    const int ny,
    const int nf,
    const int lt,
    const int rt,
    const int dn,
    const int up,

    CL_MEM_GLOBAL float * V,
    int numVertices,
    float * verticesV,
    float * verticesA,
    float slopeNegInf,
    float slopePosInf,
    int num_channels,
    CL_MEM_GLOBAL float * GSynHead,
    CL_MEM_GLOBAL float * activity)
{
   if (num_channels==1) {
      applyGSyn_HyPerLayer1Channel(numNeurons, V, GSynHead);
   }
   else {
      applyGSyn_HyPerLayer(numNeurons, V, GSynHead);
   }
   setActivity_PtwiseLinearTransferLayer(numNeurons, activity, V, nx, ny, nf, lt, rt, dn, up, numVertices, verticesA, verticesV, slopeNegInf, slopePosInf);
}
