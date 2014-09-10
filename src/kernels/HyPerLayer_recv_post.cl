//Defining a compiler directive to tell it the kernel needs the file
/* Data type for weights and activity */
#define pvwdata_t float
#define pvadata_t float
#define pvdata_t float
#define RESTRICT

//#define CL_KERNEL_INCLUDE
//#include "../layers/accumulate_functions.h"
//#undef CL_KERNEL_INCLUDE


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

//TODO see if you can include conversions.h 
static inline int kxPos(int k, int nx, int ny, int nf)
{
   return (k/nf) % nx;
}

static inline int kyPos(int k, int nx, int ny, int nf)
{
   return k / (nx*nf);
}

static inline int featureIndex(int k, int nx, int ny, int nf)
{
   return k % nf;
}

static inline int kIndex(int kx, int ky, int kf, int nx, int ny, int nf)
{
   return kf + (kx + ky * nx) * nf;
}

static inline int kIndexExtended(int k, int nx, int ny, int nf, int lt, int rt, int dn, int up)
{
   const int kx_ex = lt + kxPos(k, nx, ny, nf);
   const int ky_ex = up + kyPos(k, nx, ny, nf);
   const int kf = featureIndex(k, nx, ny, nf);
   return kIndex(kx_ex, ky_ex, kf, nx + lt + rt, ny + dn + up, nf);
}

int pvpatch_accumulate_from_post(int nk, CL_MEM_LOCAL float * v, CL_MEM_LOCAL float * a, CL_MEM_LOCAL float * w, float dt_factor, void * auxPtr) {
   //See what size nk is, and see if you can split it into vector multiplication
   int k = 0;
   while(k < nk){
      //GPUs can do 2, 4, 8, and 16 vector multiplications
      if(nk - k > 16){
         CL_MEM_LOCAL float16 * activity16 = (CL_MEM_LOCAL float16*) (a+k);
         CL_MEM_LOCAL float16 * weight16   = (CL_MEM_LOCAL float16*) (w+k);
         float16 ans16 = *activity16 * *weight16;
         *v += (ans16.s0 + ans16.s1 + ans16.s2 + ans16.s3 + ans16.s4 + ans16.s5 + ans16.s6 + ans16.s7 + 
                ans16.s8 + ans16.s9 + ans16.sa + ans16.sb + ans16.sc + ans16.sd + ans16.se + ans16.sf) * dt_factor;
         k += 16;
      }
      else if(nk - k > 8){
         CL_MEM_LOCAL float8 * activity8 = (CL_MEM_LOCAL float8*) (a+k);
         CL_MEM_LOCAL float8 * weight8   = (CL_MEM_LOCAL float8*) (w+k);
         float8 ans8 = *activity8 * *weight8;
         *v += (ans8.s0 + ans8.s1 + ans8.s2 + ans8.s3 + ans8.s4 + ans8.s5 + ans8.s6 + ans8.s7) * dt_factor;
         k += 8;
      }
      else if(nk - k > 4){
         CL_MEM_LOCAL float4 * activity4 = (CL_MEM_LOCAL float4*) (a+k);
         CL_MEM_LOCAL float4 * weight4   = (CL_MEM_LOCAL float4*) (w+k);
         float4 ans4 = *activity4 * *weight4;
         *v += (ans4.x + ans4.y + ans4.z + ans4.w) * dt_factor;
         k += 4;
      }
      else if(nk - k > 2){ 
         CL_MEM_LOCAL float2 * activity2 = (CL_MEM_LOCAL float2*) (a+k);
         CL_MEM_LOCAL float2 * weight2   = (CL_MEM_LOCAL float2*) (w+k);
         float2 ans2 = *activity2 * *weight2;
         *v += (ans2.x + ans2.y) * dt_factor;
         k += 2;
      }
      else{
         *v += a[k] * w[k] * dt_factor;
         k+= 1;
      }
   }
   return 0;
}
#endif

CL_KERNEL
void HyPerLayer_recv_post(
      const int nxRes, //num post neurons
      const int nyRes,
      const int nf,

      const int nblt, //Border of orig
      const int nbrt, //Border of orig
      const int nbdn, //Border of orig
      const int nbup, //Border of orig

      const int nxp,
      const int nyp,
      const int nfp,

      const int localBufSizeX,
      const int localBufSizeY,
      const float preToPostScaleX,
      const float preToPostScaleY,

      const int sy,
      const int syp,
      const int numPerStride,
      const float dt_factor,
      const int sharedWeights,

      CL_MEM_GLOBAL long* startSourceExtBuf,
      CL_MEM_GLOBAL float* preData,
      CL_MEM_GLOBAL float* weights,
      CL_MEM_GLOBAL float* postGsyn,
      CL_MEM_GLOBAL int* patch2datalookuptable,

      CL_MEM_LOCAL float* preBuffer,
      CL_MEM_LOCAL float* postBuffer,
      CL_MEM_LOCAL float* weightsBuffer
){
#ifdef PV_USE_OPENCL
      int fTargetRes = get_global_id(0);
      int xTargetRes = get_global_id(1);
      int yTargetRes = get_global_id(2);

      int localF = get_local_size(0);
      int localX = get_local_size(1);
      int localY = get_local_size(2);
      
      int localFIndex = get_local_id(0);
      int localXIndex = get_local_id(1);
      int localYIndex = get_local_id(2);

      //Calculate kTargetRes based on x, y, and f
      int kTargetRes = kIndex(xTargetRes, yTargetRes, fTargetRes, nxRes, nyRes, nf);
      //Change restricted to extended post neuron
      int kTargetExt = kIndexExtended(kTargetRes, nxRes, nyRes, nf, nblt, nbrt, nbdn, nbup);

      //TODO this won't work with HyPerConns
      CL_MEM_LOCAL int wIdx;
      if(localXIndex == 0 && localYIndex == 0){
         int kernelIndex;
         if(sharedWeights == 1){
            kernelIndex = patch2datalookuptable[kTargetExt];
         }
         else{
            kernelIndex = kTargetExt;
         }
         wIdx = kernelIndex * nxp * nyp * nfp;
      }

      //Get top left most neuron in the group
      CL_MEM_LOCAL long localStartSourceExt;
      if(localXIndex == 0 && localYIndex == 0 && localFIndex == 0){
         localStartSourceExt = startSourceExtBuf[kTargetRes];
      }

      int localIndex = kIndex(localXIndex, localYIndex, localFIndex, localX, localY, localF);

      postBuffer[localIndex] = 0;

      int numXfBuffer = localBufSizeX * nfp;

      //Match preBuffer to indPreData, need to find x and y offsets
      int xOffset = localXIndex * preToPostScaleX;
      //int yOffset = localYIndex * preToPostScaleY;

      int numThreads = localX * localY * localF;

      CL_MEM_LOCAL float* postAddr = postBuffer + localIndex;
      
      //All processes in workgroup needs that local index
      barrier(CLK_LOCAL_MEM_FENCE);
      
      for(int ky = 0; ky < nyp; ky++){
         //Copy global to local, do this with all threads
         //Pre buffer
         async_work_group_copy(preBuffer, &(preData[localStartSourceExt + ky * sy]), numXfBuffer, 0);
         //for(int i = localIndex; i < numXfBuffer; i+=numThreads){
         //   preBuffer[i] = preData[localStartSourceExt + ky * sy + i];
         //}
         //Weights
         async_work_group_copy(weightsBuffer, &weights[wIdx+ky * syp], nxp*nfp, 0);
         //for(int i = localIndex; i < nxp*nfp; i+= numThreads){
         //   weightsBuffer[i] = weights[wIdx + ky * syp + i];
         //}
         barrier(CLK_LOCAL_MEM_FENCE);
         
         CL_MEM_LOCAL float * activityY = &(preBuffer[xOffset * nfp]);
         //CL_MEM_GLOBAL float * activityY = &(preData[startSourceExt + ky * sy]);
         //CL_MEM_GLOBAL float * weightY = &(weights[wIdx + ky*syp]);
         //Summing into post buffer indexed by localIndex
         //pvpatch_accumulate_from_post(numPerStride, postAddr, activityY, weightY, dt_factor, (void*)0);
         pvpatch_accumulate_from_post(numPerStride, postAddr, activityY, weightsBuffer, dt_factor, (void*)0);
         barrier(CLK_LOCAL_MEM_FENCE);
      }

      //Barrier to make sure the work group's postbuffer is set
      barrier(CLK_LOCAL_MEM_FENCE);

      //Sum into global memory
      postGsyn[kTargetRes] += postBuffer[localIndex];
#endif
}
