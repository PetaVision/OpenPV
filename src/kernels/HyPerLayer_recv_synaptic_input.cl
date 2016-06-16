
#ifndef PV_USE_OPENCL
#  define CL_KERNEL
#  define CL_MEM_GLOBAL
#  define CL_MEM_CONST
#  define CL_MEM_LOCAL
#  define PV_OPENCL_PRINTF pvInfo().printf
#  include <stdio.h>
#  include "utils/PVLog.hpp"
#  include "utils/PVAssert.hpp"
#else  /* compiling with OpenCL */
#  pragma OPENCL EXTENSION cl_khr_int32_base_atomics : enable
#  pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#  pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#  pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#  pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable
#  pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#  define CL_KERNEL       __kernel
#  define CL_MEM_GLOBAL   __global
#  define CL_MEM_CONST    __constant
#  define CL_MEM_LOCAL    __local
#  define PV_OPENCL_PRINTF printf
#endif



//#define KX  get_global_id(0)
//#define KY  get_global_id(1)
#define KX  get_group_id(0)
#define KY  get_group_id(1)
#define KXL get_local_id(0)
#define KYL get_local_id(1)

#define NX  get_group_size(0)
#define NY  get_group_size(1)
#define NXL  get_local_size(0)
#define NYL  get_local_size(1)

#ifdef PV_USE_OPENCL
//static inline void getSemaphorLoc(CL_MEM_LOCAL int * semaphor) {
//   while(atom_cmpxchg(semaphor, 0, 1)!=0);
////   int occupied = atom_xchg(semaphor, 1);
////   while(occupied > 0)
////   {
////     occupied = atom_xchg(semaphor, 1);
////   }
//}
//static inline void releaseSemaphorLoc(CL_MEM_LOCAL int * semaphor) {
//   atom_xchg(semaphor, 0);
//}
//static inline void getSemaphorGL(CL_MEM_GLOBAL int * semaphor) {
//   while(atom_cmpxchg(semaphor, 0, 1)!=0);
////   int occupied = atom_xchg(semaphor, 1);
////   while(occupied > 0)
////   {
////     occupied = atom_xchg(semaphor, 1);
////   }
//}
//static inline void releaseSemaphorGL(CL_MEM_GLOBAL int * semaphor) {
//   atom_xchg(semaphor, 0);
//}
inline void AtomicAddGL(volatile CL_MEM_GLOBAL float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile CL_MEM_GLOBAL unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}
inline void AtomicAddLOC(volatile CL_MEM_LOCAL float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile CL_MEM_LOCAL unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}
#endif



/**
 * update post-synaptic conductance based on pre-synaptic activity
 */
CL_KERNEL void HyPerLayer_recv_synaptic_input (
#ifndef PV_USE_OPENCL
      int kx, int ky, int lidx, int lidy, int nxl, int nyl,
#endif
          int nxPre,
          int nyPre,
          int nfPre,
          int nbPre,
          int nxp,
          int nyp,
          int nfp,
          float fScale,
          float xScale,
          float yScale,
          size_t offsetA,
          CL_MEM_GLOBAL int * p2dLUT,
          CL_MEM_GLOBAL float * A,
          CL_MEM_GLOBAL float * W,
          int Gstart,   // must be an integer for atomic updates
          CL_MEM_GLOBAL float   *G   // must be an integer for atomic updates
       )
{
#ifdef PV_USE_OPENCL
      int kx,  ky,  lidx,  lidy,  nxl,  nyl;
      kx=KX*NXL+NXL/2; //(nxPre+2*nbPre)/2;
      ky=KY*NYL+NYL/2; //(nyPre+2*nbPre)/2;
      lidx=KXL;
      lidy=KYL;
      nxl=NXL;
      nyl=NYL;
#endif
   // This kernel is to be run over the extended pre-synaptic layer
   // The easiest way to connect to post-synaptic layer is to shift to
   // non-extended coordinates and then to scale the results
   // WARNING, here we assume # post-synaptic layer >= pre-synaptic #


   int kxl=lidx+kx-nxl/2;
   int kyl=lidy+ky-nyl/2;
   int kPre=kyl*(nxPre+2*nbPre)*nfPre+kxl;
   int kPreW=kPre;
   if(p2dLUT[0]!=-1) kPreW=p2dLUT[kPre];
   int wPatchSize = nxp*nyp*nfp;
   int wOffset=wPatchSize*kPreW;

#ifdef PV_USE_OPENCL
   float activity=A[kPre+offsetA];
#else
   float activity=A[kPre];
#endif
   if (activity > 0.0) {

   //const int kPostXFL = (int)(fScale*xScale*((float)kxl)-fScale*xScale*((float)nbPre)); // kPostX==0 is left boundary non-extended
   const int kPreXL = (int)(kxl/nfPre); // kPostX==0 is left boundary non-extended
   const int kPostXL = floor(xScale*((float)(kPreXL-nbPre))); // kPostX==0 is left boundary non-extended
   const int kPostYL = floor(yScale*((float)(kyl-nbPre))); // kPostY==0 is top  boundary non-extended
   // keep within post-synaptic, non-extended boundary
   //
   const int gStride = (int)(xScale*fScale*((float)nxPre*nfPre));

   //const int gxl=kPostXL-(int)(((float)(nxp))/2.0f-0.5f);
   //const int gyl=kPostYL-(int)(((float)(nyp))/2.0f-0.5f);
   int gxl=kPostXL;
   int gyl=kPostYL;

   if(xScale>1){
      gxl-=(nxp/xScale-1)/2;
   }
   else {
      gxl-=(nxp-1)/2;
   }
   if(yScale>1){
      gyl-=(nyp/yScale-1)/2;
   }
   else {
      gyl-=(nyp-1)/2;
   }

   // loop over weight patch updating G atomically
   for (int j = 0; j < nyp; j++) {
      int gAy=gyl+j;
      if((gAy>=0)&&(gAy<(int)(yScale*nyPre))) {
         for (int i = 0; i < nxp*nfp; i++) {
            int gAxlf=gxl*nfp+i;
            if((gAxlf>=0)&&(gAxlf<(int)(xScale*fScale*((float)nxPre*nfPre)))) {
               int gAddy=Gstart+(gAy)*gStride + gAxlf;
               int weightptr=j*nxp*nfp+i;
               float answer=activity * W[weightptr+wOffset];
#ifndef PV_USE_OPENCL
               G[gAddy] += answer;
               if(0) {
                  PV_OPENCL_PRINTF ("j %d\n",j);
                  PV_OPENCL_PRINTF ("i %d\n",i);
                  PV_OPENCL_PRINTF ("kxl %d\n",kxl);
                  PV_OPENCL_PRINTF ("kyl %d\n",kyl);
                  PV_OPENCL_PRINTF ("kx %d\n",kx);
                  PV_OPENCL_PRINTF ("ky %d\n",ky);
                  PV_OPENCL_PRINTF ("lidx %d\n",lidx);
                  PV_OPENCL_PRINTF ("lidy %d\n",lidy);
                  PV_OPENCL_PRINTF ("nxp %d\n",nxp);
                  PV_OPENCL_PRINTF ("nyp %d\n",nyp);
                  PV_OPENCL_PRINTF ("nx %d\n",nxl);
                  PV_OPENCL_PRINTF ("nyl %d\n",nyl);
                  PV_OPENCL_PRINTF ("activity %f\n",activity);
                  PV_OPENCL_PRINTF ("W[weightptr+wOffset] %f\n",W[weightptr+wOffset]);
               }
#else
               if(answer!=0) AtomicAddGL(&G[gAddy], answer);
#endif
            } // if gAxlf in bounds
         } //for nxp*nfp
      } //if gAy in bounds
   }  // end nyp loop
}     // end if activity

}
