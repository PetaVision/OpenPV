
#ifndef PV_USE_OPENCL
#  define CL_KERNEL
#  define CL_MEM_GLOBAL
#  define CL_MEM_CONST
#  define CL_MEM_LOCAL
#  include <stdio.h>
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
#else
          //volatile CL_MEM_LOCAL int *gTempSemaphor,
#endif
          //CL_MEM_LOCAL float *gtemp,
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
          CL_MEM_GLOBAL float * A,
          CL_MEM_GLOBAL float * W,
#ifdef PV_USE_OPENCL
          //volatile CL_MEM_GLOBAL int *gSemaphor,
#endif
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
//      CL_MEM_LOCAL int *gtemp;
#endif
   // This kernel is to be run over the extended pre-synaptic layer
   // The easiest way to connect to post-synaptic layer is to shift to
   // non-extended coordinates and then to scale the results
   // WARNING, here we assume # post-synaptic layer >= pre-synaptic #


   int kxl=lidx+kx-nxl/2;
   int kyl=lidy+ky-nyl/2;
   int kPre=kyl*(nxPre*nfPre+2*nbPre)+kxl;
   int wPatchSize = nxp*nyp*nfp;
   int wOffset=wPatchSize*kPre;
   //int tempBufStride=nxl+nxp*nfp;
   
   const int tempXPatchSize = (10*(nxp*nfp+nxl)/nxl+9)/10;
   const int tempYPatchSize = (10*(nyp+nyl)/nyl+9)/10;

//#ifdef PV_USE_OPENCL
//   for(int xcpl=tempXPatchSize*lidx; (xcpl<tempXPatchSize*(lidx+1))&&(xcpl<nxp*nfp+nxl); xcpl++) {
//      for(int ycpl=tempYPatchSize*lidy; (ycpl<tempYPatchSize*(lidy+1))&&(ycpl<nyp+nyl); ycpl++) {
//         gtemp[ycpl*tempBufStride + xcpl]=0;
//         //gTempSemaphor[ycpl*tempBufStride + xcpl]=0;
//      }
//   }
//   barrier(CLK_LOCAL_MEM_FENCE);
//#endif

#ifdef PV_USE_OPENCL
   float activity=A[kPre+offsetA];
#else
   float activity=A[kPre];
#endif
   if (activity > 0.0) {

      const int kPostXL = (int)(xScale*kxl) - (int)(xScale*nbPre); // kPostX==0 is left boundary non-extended
      const int kPostYL = (int)(yScale*kyl) - (int)(yScale*nbPre); // kPostY==0 is top  boundary non-extended
      // keep within post-synaptic, non-extended boundary
      //
      const int gStride = xScale*nxPre*nfPre;

      const int gxl=kPostXL-nxp*nfp/2;
      const int gyl=kPostYL-nyp/2;
   // loop over weight patch updating G atomically
   for (int j = 0; j < nyp; j++) {
      //int temp_idy=j+lidy;
      int gAy=gyl+j;
      if((gAy>=0)&&(gAy<(int)(yScale*nyPre))) {
      //if((j+kyl>0)&&(j+kyl<nyPre+2*nbPre)) {
         for (int i = 0; i < nxp*nfp; i++) {
            int gAx=gxl+i;
            if((gAx>=0)&&(gAx<(int)(xScale*nxPre*nfPre))) {
            //if((i+kxl>0)&&(i+kxl<nxPre*nfPre+2*nbPre)) {
               int gAddy=(gAy)*gStride + gAx;
               //int temp_idx=i+lidx;
               //int tempid=temp_idy*tempBufStride+temp_idx;
               int weightptr=j*nxp*nfp+i;
               float answer=activity * W[weightptr+wOffset];
#ifndef PV_USE_OPENCL
               //gtemp[tempid] += answer;
               if((kx<10)&&(ky<10)&&(kxl<5)&&(kyl<5)&&0) {
                  printf("tempid %d\n",tempid);
                  printf("temp_idy %d\n",temp_idy);
                  printf("temp_idx %d\n",temp_idx);
                  printf("j %d\n",j);
                  printf("i %d\n",i);
                  printf("kxl %d\n",kxl);
                  printf("kyl %d\n",kyl);
                  printf("kx %d\n",kx);
                  printf("ky %d\n",ky);
                  printf("lidx %d\n",lidx);
                  printf("lidy %d\n",lidy);
                  printf("nxp %d\n",nxp);
                  printf("nyp %d\n",nyp);
                  printf("nx %d\n",nxl);
                  printf("nyl %d\n",nyl);
                  printf("activity %f\n",activity);
                  printf("W[weightptr+wOffset] %f\n",W[weightptr+wOffset]);
                  printf("tempVal %f\n",tempVal);
                  //printf("gtemp[tempid] %f\n",gtemp[tempid]);
               }
#else
               if(answer!=0) AtomicAddGL(&G[gAddy], answer);
               //if(tempVal!=0) AtomicAddLOC(&gtemp[tempid], tempVal);
#endif
            }
         }
      }
   }  // end nyp loop
}     // end if activity

   //copy back to G:
//   const int kPostX = (int)(xScale*kx) - (int)(xScale*nbPre); // kPostX==0 is left boundary non-extended
//   const int kPostY = (int)(yScale*ky) - (int)(yScale*nbPre); // kPostY==0 is top  boundary non-extended
//   // keep within post-synaptic, non-extended boundary
//   //
//   const int gStride = xScale*nxPre*nfPre;
//
//   const int gx=kPostX-nxl/2-nxp*nfp/2;
//   const int gy=kPostY-nyl/2-nyp/2;
//#ifdef PV_USE_OPENCL
//   barrier(CLK_LOCAL_MEM_FENCE);
//#endif
//   for(int xcpl=tempXPatchSize*lidx; (xcpl<tempXPatchSize*(lidx+1)&&(xcpl<nxp*nfp+nxl)); xcpl++) {
//      int gAx=gx+xcpl;
//      if((gAx>=0)&&(gAx<(int)(xScale*nxPre*nfPre))) {
//         for(int ycpl=tempYPatchSize*lidy; (ycpl<tempYPatchSize*(lidy+1))&&(ycpl<nyp+nyl); ycpl++) {
//            int gAy=gy+ycpl;
//            if((gAy>=0)&&(gAy<(int)(yScale*nyPre))) {
//               int gAddy=(gAy)*gStride + gAx;
//               float answer=gtemp[ycpl*tempBufStride + xcpl];
//#ifdef PV_USE_OPENCL
//               if(answer!=0) AtomicAddGL(&G[gAddy], answer);
//#else
//                if((G[gAddy]<25)&&0) {
//                  printf("gAddy %d\n",gAddy);
//                  printf("gAy %d\n",gAy);
//                  printf("gAx %d\n",gAx);
//                  printf("ycpl %d\n",ycpl);
//                  printf("xcpl %d\n",xcpl);
//                  printf("kxl %d\n",kxl);
//                  printf("kyl %d\n",kyl);
//                  printf("kx %d\n",kx);
//                  printf("ky %d\n",ky);
//                  printf("gx %d\n",gx);
//                  printf("gy %d\n",gy);
//                  printf("(int)(xScale*nxPre*nfPre) %d\n",(int)(xScale*nxPre*nfPre));
//                  printf("(int)(yScale*nyPre) %d\n",(int)(yScale*nyPre));
//                  printf("lidx %d\n",lidx);
//                  printf("lidy %d\n",lidy);
//                  printf("nxp %d\n",nxp);
//                  printf("nyp %d\n",nyp);
//                  printf("nx %d\n",nxl);
//                  printf("nyl %d\n",nyl);
//                  printf("answer %f\n",answer);
//                  printf("G[gAddy] %f\n",G[gAddy]);
//                  printf("gtemp[ycpl*tempBufStride + xcpl] %f\n",gtemp[ycpl*tempBufStride + xcpl]);
//               }
//#endif
//            }
//         }
//      }
//   }
//#ifndef PV_USE_OPENCL
//   for(int clidx=0;clidx<nxl+nxp*nfp;clidx++){
//      int gAx=gx+clidx;
//      if((gAx>=0)&&(gAx<(int)(xScale*nxPre*nfPre))) {
//         for(int clidy=0;clidy<nyl+nyp;clidy++){
//            int gAy=gy+clidy;
//            if((gAy>=0)&&(gAy<(int)(yScale*nyPre))) {
//               unsigned int gAddy=(gAy)*gStride + gAx;
//               double answer=gtemp[clidy*tempBufStride + clidx];
//               G[gAddy]+=answer;
//            }
//         }
//      }
//   }
//#endif
}
