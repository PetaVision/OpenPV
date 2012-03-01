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
#  include "conversions.hcl"
#endif


//
// update the state of a retinal layer (spiking)
//
//    assume called with 1D kernel
//
CL_KERNEL
void ANNLayer_update_state(
    const int nx,
    const int ny,
    const int nf,
    const int nb,

    CL_MEM_GLOBAL float * V,
    const float Vth,
    const float VMax,
    const float VMin,
    CL_MEM_GLOBAL float * GSynExc,
    CL_MEM_GLOBAL float * GSynInh,
    CL_MEM_GLOBAL float * activity)
{
   int k;

#ifndef PV_USE_OPENCL

for (k = 0; k < nx*ny*nf; k++) {
#else
   k = get_global_id(0);
#endif

   int kex = kIndexExtended(k, nx, ny, nf, nb);

   //
   // kernel (nonheader part) begins here
   //


   // local variables
   float l_activ;
   float l_V   = V[k];

   float l_GSynExc  = GSynExc[k];
   float l_GSynInh  = GSynInh[k];

   //updateV():
   l_V=l_GSynExc-l_GSynInh;
   //applyVMax():
   if(l_V > VMax) l_V = VMax;
   //applyVThresh():
   if(l_V < Vth) l_V = VMin;

   //setActivity():
   l_activ=l_V;


   activity[kex] = l_activ;
   V[k]   = l_V;

   //resetGSynBuffers():
   GSynExc[k]  = 0.0f;
   GSynInh[k]  = 0.0f;

#ifndef PV_USE_OPENCL
   } // loop over k
#endif

}
