#include "../layers/updateStateFunctions.h"
#include <stdbool.h>


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
// update the state of an HyPerLCA layer
//
// To allow porting to GPUs, functions called from this function must be
// static inline functions.  If a subclass needs new behavior, it needs to
// have its own static inline function.
//
CL_KERNEL
void HyPerLCALayer_update_state(
    const int nbatch,
    const int numNeurons,
    const int nx,
    const int ny,
    const int nf,
    const int lt,
    const int rt,
    const int dn,
    const int up,
    const int numChannels,

    CL_MEM_GLOBAL float * V,
    int numVertices,
    float * verticesV,
    float * verticesA,
    float * slopes,
    const bool selfInteract,
    CL_MEM_GLOBAL double* dtAdapt,
    const float tau,
    CL_MEM_GLOBAL float * GSynHead,
    CL_MEM_GLOBAL float * activity)
{
   updateV_HyPerLCALayer(nbatch, numNeurons, numChannels, V, GSynHead, activity,
		   numVertices, verticesV, verticesA, slopes, dtAdapt, tau, selfInteract, nx, ny, nf, lt, rt, dn, up);
}
