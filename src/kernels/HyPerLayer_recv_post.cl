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

static inline int kIndexExtended(int k, int nx, int ny, int nf, int nb)
{
   const int kx_ex = nb + kxPos(k, nx, ny, nf);
   const int ky_ex = nb + kyPos(k, nx, ny, nf);
   const int kf = featureIndex(k, nx, ny, nf);
   return kIndex(kx_ex, ky_ex, kf, nx + 2*nb, ny + 2*nb, nf);
}

int pvpatch_accumulate_from_post(int nk, CL_MEM_LOCAL float * v, CL_MEM_LOCAL float * a, CL_MEM_GLOBAL float * w, float dt_factor, void * auxPtr) {
   //See what size nk is, and see if you can split it into vector multiplication
   int k = 0;
   while(k < nk){
      //GPUs can do 2, 4, 8, and 16 vector multiplications
      if(nk - k > 16){
         float16 activity16 = (float16)(a[k], a[k+1], a[k+2], a[k+3], a[k+4], a[k+5], a[k+6], a[k+7],
                                        a[k+8], a[k+9], a[k+10], a[k+11], a[k+12], a[k+13], a[k+14], a[k+15]);
         float16 weight16 = (float16)(w[k], w[k+1], w[k+2], w[k+3], w[k+4], w[k+5], w[k+6], w[k+7],
                                      w[k+8], w[k+9], w[k+10], w[k+11], w[k+12], w[k+13], w[k+14], w[k+15]);
         float16 ans16 = activity16 * weight16;
         *v += (ans16.s0 + ans16.s1 + ans16.s2 + ans16.s3 + ans16.s4 + ans16.s5 + ans16.s6 + ans16.s7 + 
                ans16.s8 + ans16.s9 + ans16.sa + ans16.sb + ans16.sc + ans16.sd + ans16.se + ans16.sf) * dt_factor;
         k += 16;
      }
      else if(nk - k > 8){
         float8 activity8 = (float8)(a[k], a[k+1], a[k+2], a[k+3], a[k+4], a[k+5], a[k+6], a[k+7]);
         float8 weight8 = (float8)(w[k], w[k+1], w[k+2], w[k+3], w[k+4], w[k+5], w[k+6], w[k+7]);
         float8 ans8 = activity8 * weight8;
         *v += (ans8.s0 + ans8.s1 + ans8.s2 + ans8.s3 + ans8.s4 + ans8.s5 + ans8.s6 + ans8.s7) * dt_factor;
         k += 8;
      }
      else if(nk - k > 4){
         float4 activity4 = (float4)(a[k], a[k+1], a[k+2], a[k+3]);
         float4 weight4 = (float4)(w[k], w[k+1], w[k+2], w[k+3]);
         float4 ans4 = activity4 * weight4;
         *v += (ans4.x + ans4.y + ans4.z + ans4.w) * dt_factor;
         k += 4;
      }
      else if(nk - k > 2){ 
         float2 activity2 = (float2)(a[k], a[k+1]);
         float2 weight2 = (float2)(w[k], w[k+1]);
         float2 ans2 = activity2 * weight2;
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
      const int nb, //Border of orig
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
      CL_MEM_LOCAL float* postBuffer
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
      int kTargetExt = kIndexExtended(kTargetRes, nxRes, nyRes, nf, nb);

      //Get top left most neuron in the group
      CL_MEM_LOCAL long localStartSourceExt;
      if(localXIndex == 0 && localYIndex == 0 && localFIndex == 0){
         localStartSourceExt = startSourceExtBuf[kTargetRes];
      }
      //All processes in workgroup needs that local index
      barrier(CLK_LOCAL_MEM_FENCE);
      
      //Move global predata to buffer
      //Find number of global threads working in this work group
      int numLocal = localX * localY * localF;
      //Find total number of pre neurons needed in the buffer
      int totPreBuf = localBufSizeX * localBufSizeY * nfp;
      //Find a good splitting number, using ceil to make sure we get everything
      int numPrePerLocal = ceil((float)totPreBuf/(float)numLocal);

      int localIndex = kIndex(localXIndex, localYIndex, localFIndex, localX, localY, localF);
      //Set up pre
      for(int i = 0; i < numPrePerLocal; i++){
         //Need to get a mapping from local index (post) to pre index
         int mappedPreIndex = localIndex * numPrePerLocal + i;
         if(mappedPreIndex < totPreBuf){
            //Convert mappedPreIndex into a pre data index
            int preIdxX = kxPos(mappedPreIndex, localBufSizeX, localBufSizeY, nfp);
            int preIdxY = kyPos(mappedPreIndex, localBufSizeX, localBufSizeY, nfp); 
            int preIdxF = featureIndex(mappedPreIndex, localBufSizeX, localBufSizeY, nfp);
            //Convert mappedPreIndex into a preBufIdx
            int preBufIdx = kIndex(preIdxX, preIdxY, preIdxF, localBufSizeX, localBufSizeY, nfp);
            //X, Y, and F are with respect to the buffer size
            int xfIdx = preIdxX * nfp + preIdxF;
            //Using the orig stride, we should be able to go to the correct row in the data
            preBuffer[preBufIdx] = preData[localStartSourceExt + preIdxY * sy + xfIdx];
         }
      }

      //Initialize post
      postBuffer[localIndex] = 0;

      //Barrier to make sure the work group's prebuffer is set
      barrier(CLK_LOCAL_MEM_FENCE);

      int kernelIndex;
      if(sharedWeights == 1){
         kernelIndex = patch2datalookuptable[kTargetExt];
      }
      else{
         kernelIndex = kTargetExt;
      }
      int wIdx = kernelIndex * nxp * nyp * nfp;

      //Match preBuffer to indPreData, need to find x and y offsets
      int xOffset = localXIndex * preToPostScaleX;
      int yOffset = localYIndex * preToPostScaleY;

      //Find post buffer address
      CL_MEM_LOCAL float * postAddr = postBuffer + localIndex;

      for(int ky = 0; ky < nyp; ky++){
         CL_MEM_LOCAL float * activityY = &(preBuffer[(ky+yOffset) * localBufSizeX * nfp + xOffset*nfp]);
         CL_MEM_GLOBAL float * weightY = &(weights[wIdx + ky*syp]);
         //Summing into post buffer indexed by localIndex
         pvpatch_accumulate_from_post(numPerStride, postAddr, activityY, weightY, dt_factor, (void*)0);
      }

      //Barrier to make sure the work group's postbuffer is set
      barrier(CLK_LOCAL_MEM_FENCE);

      //Sum into global memory
      CL_MEM_GLOBAL pvdata_t * gSynPatchPos = postGsyn + kTargetRes;
      *gSynPatchPos += postBuffer[localIndex];


#endif
}
