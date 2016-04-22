//Defining a compiler directive to tell it the kernel needs the file
/* Data type for weights and activity */
#include "../include/pv_datatypes.h"
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

typedef struct PVPatch_ {
   unsigned int offset;
   unsigned short nx, ny;
} PVPatch;


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

int pvpatch_accumulate(int nk, CL_MEM_GLOBAL float * v, float a, CL_MEM_GLOBAL float * w, void * auxPtr) {
   int k = 0;
   while(k < nk){
      //GPUs can do 2, 4, 8, and 16 vector multiplications
      if(nk - k > 16){
         float16 a16 = (float16)(a, a, a, a, a, a, a, a,
                                 a, a, a, a, a, a, a, a);
         
         float16 w16 = (float16)(w[k], w[k+1], w[k+2], w[k+3], w[k+4], w[k+5], w[k+6], w[k+7],
                                      w[k+8], w[k+9], w[k+10], w[k+11], w[k+12], w[k+13], w[k+14], w[k+15]);
         CL_MEM_GLOBAL float16* v16 = (CL_MEM_GLOBAL float16*) (v+k);
         *v16 += a16 * w16;
         
         k += 16;
      }
      else if(nk - k > 8){
         float8 a8 = (float8)(a, a, a, a, a, a, a, a);
         float8 w8 = (float8)(w[k], w[k+1], w[k+2], w[k+3], w[k+4], w[k+5], w[k+6], w[k+7]);
         CL_MEM_GLOBAL float8* v8 = (CL_MEM_GLOBAL float8*)(v+k);
         *v8 += a8 * w8;
         k += 8;
      }
      else if(nk - k > 4){
         float4 a4 = (float4)(a, a, a, a);
         float4 w4 = (float4)(w[k], w[k+1], w[k+2], w[k+3]);
         CL_MEM_GLOBAL float4* v4 = (CL_MEM_GLOBAL float4*)(v+k);
         *v4 += a4 * w4;
         k += 4;
      }
      else if(nk - k > 2){ 
         float2 a2 = (float2)(a, a);
         float2 w2 = (float2)(w[k], w[k+1]);
         CL_MEM_GLOBAL float2* v2 = (CL_MEM_GLOBAL float2*)(v+k);
         *v2 += a2 * w2;
         k += 2;
      }
      else{
         v[k] += a * w[k];
         k+= 1;
      }
   }
   return 0;
}
#endif

CL_KERNEL
void HyPerLayer_recv_pre(
      const int preNxExt,
      const int preNyExt,
      const int preNf,
      const int postNxRes,
      const int postNyRes,
      const int postNf,

      const int nxp,
      const int nyp,
      const int nfp,
      const int groupXSize,
      const int groupYSize,
      const int localPreSizeX,
      const int localPreSizeY,
      const int localBufSizeX,
      const int localBufSizeY,

      const int sy,
      const int syw,
      const float dt_factor,
      const int sharedWeights,

      CL_MEM_CONST PVPatch* patches,
      CL_MEM_CONST size_t* gSynPatchStart,

      CL_MEM_GLOBAL long* postToPreActivity,
      CL_MEM_GLOBAL float* preData,
      CL_MEM_GLOBAL float* weights,
      CL_MEM_GLOBAL float* postGSyn,
      CL_MEM_GLOBAL int* patch2datalookuptable
){
#ifdef PV_USE_OPENCL
      int groupXPost = get_global_id(0);
      int groupYPost = get_global_id(1);
      int numXGroups = get_global_size(0);
      int numYGroups = get_global_size(1);

      int localX = get_local_size(0);
      int localY = get_local_size(1);

      int localXIndex = get_local_id(0);
      int localYIndex = get_local_id(1);

      //Find top left post index based on group index to copy activity data into local

      int postX = groupXPost * groupXSize;
      int postY = groupYPost * groupYSize;
      int kPostRes = kIndex(postX, postY, 0, postNxRes, postNyRes, postNf); 

      //CL_MEM_LOCAL int groupPostRes;
      //CL_MEM_LOCAL long groupStartPreExt;
      //if(localXIndex == 0 && localYIndex == 0){
      //   groupPostRes = kPostRes;
      //   groupStartPreExt = postToPreActivity[groupPostRes];
      //}

      ////Clear post buffer
      //int postBufXSize = groupXSize * localX;
      //int postBufYStride = postBufXSize * postNf;
      //int postBufNumXF = groupXSize * postNf;
      //CL_MEM_LOCAL float* localGSynStart = &(postBuffer[localYIndex * groupYSize * postBufYStride
      //                                                              + localXIndex * groupXSize * postNf]);

      //for(int ky = 0; ky < groupYSize; ky++){
      //   CL_MEM_LOCAL float* gSynStartY = localGSynStart + ky * postBufYStride;
      //   for(int kxf = 0; kxf < postBufNumXF; kxf++){
      //      gSynStartY[kxf] = 0;
      //   }
      //}
      //barrier(CLK_LOCAL_MEM_FENCE);

      ////Find number of global threads working in this workgroup
      //int numLocal = localX * localY;
      ////Find total number of pre neurons in the buffer
      //int totPreBuf = localBufSizeX * localBufSizeY * preNf;
      ////Find a good splitting number, using ceil to make sure we get everything
      //int numPrePerLocal = ceil((float)totPreBuf/(float)numLocal);
      //int localIndex = kIndex(localXIndex, localYIndex, 0, localX, localY, 1);

      ////Find pre activity start from post index

      //for(int i = 0; i < numPrePerLocal; i++){
      //   //Need to get a mapping from localIndex (post group) to pre index
      //   int mappedPreIndex = localIndex * numPrePerLocal + i;
      //   if(mappedPreIndex < totPreBuf){
      //      //Convert mappedPreIndex into a pre data index
      //      int preIdxX = kxPos(mappedPreIndex, localBufSizeX, localBufSizeY, preNf);
      //      int preIdxY = kyPos(mappedPreIndex, localBufSizeX, localBufSizeY, preNf);
      //      int preIdxF = featureIndex(mappedPreIndex, localBufSizeX, localBufSizeY, preNf);

      //      //Convert mappedPreIndex into a preBufIdx
      //      int preBufIdx = kIndex(preIdxX, preIdxY, preIdxF, localBufSizeX, localBufSizeY, preNf);

      //      //X, Y, and F are with respect to the buffer size
      //      int xfIdx = preIdxX * preNf + preIdxF;

      //      int dataIdx = groupStartPreExt + preIdxY * (preNxExt * preNf) + xfIdx;
      //      //Using the orig stride, we should be able to go to the correct row in the data
      //      preBuffer[preBufIdx] = preData[dataIdx];
      //   }
      //}

      ////Barrier to make sure the work group's prebuffer is set
      //barrier(CLK_LOCAL_MEM_FENCE);

      //Calculate this local group's startPreExt
      int startPreExt = postToPreActivity[kPostRes];

      CL_MEM_GLOBAL float* gSynStart = &(postGSyn[kPostRes]);

      ////Calculate offset in x and y direction from starting group to this group
      ////We shouldn't need F
      //int kGroupPreX = kxPos(groupStartPreExt, preNxExt, preNyExt, preNf);
      //int kGroupPreY = kyPos(groupStartPreExt, preNxExt, preNyExt, preNf);
      //int kCurPreX = kxPos(startPreExt, preNxExt, preNyExt, preNf);
      //int kCurPreY = kyPos(startPreExt, preNxExt, preNyExt, preNf);
      //int diffX = kCurPreX - kGroupPreX;
      //int diffY = kCurPreY - kGroupPreY;
      //int preLocalStart = kIndex(diffX, diffY, 0, localBufSizeX, localBufSizeY, preNf);

      //Loop through pre activity
      for(int kPreLocal = 0; kPreLocal < localPreSizeY * localPreSizeX * preNf; kPreLocal++){
         //Find x, y, and f index from kPreLocal
         int kPreXLocal = kxPos(kPreLocal, localPreSizeX, localPreSizeY, preNf);
         int kPreYLocal = kyPos(kPreLocal, localPreSizeX, localPreSizeY, preNf);
         int kPreFLocal = featureIndex(kPreLocal, localPreSizeX, localPreSizeY, preNf);

         //int kPreBuf = preLocalStart + kPreYLocal * (localBufSizeX * preNf) + kPreXLocal * preNf + kPreFLocal;
         int kPre = startPreExt + kPreYLocal * (preNxExt * preNf) + kPreXLocal * preNf + kPreFLocal;

         //float a = preBuffer[kPreBuf] * dt_factor;
         float a = preData[kPre] * dt_factor;
         if(a == 0) continue;

         //GSynPatchStart is in local post space, need to translate to buffer post space
         size_t localGSynOffset = gSynPatchStart[kPreLocal];

         int localXGSynOffset = kxPos(localGSynOffset, groupXSize, groupYSize, postNf);
         int localYGSynOffset = kyPos(localGSynOffset, groupXSize, groupYSize, postNf);
         int localFGSynOffset = featureIndex(localGSynOffset, groupXSize, groupYSize, postNf);
         //size_t gSynOffset = localYGSynOffset * postBufYStride + localXGSynOffset * postNf + localFGSynOffset;
         size_t gSynOffset = localYGSynOffset * sy + localXGSynOffset * postNf + localFGSynOffset;

         //Grab weight patches
         PVPatch patch = patches[kPreLocal];
         int nk = nfp * patch.nx;
         int ny = patch.ny;

         int kernelIndex;
         if(sharedWeights == 1){
            kernelIndex = patch2datalookuptable[kPre];
         }
         else{
            kernelIndex = kPre;
         }

         int wIdx = kernelIndex * nxp * nyp * nfp + patch.offset;

         for(int ky = 0; ky < ny; ky++){
            //CL_MEM_LOCAL float * gSynY = localGSynStart + gSynOffset + ky * postBufYStride;
            CL_MEM_GLOBAL float * gSynY = gSynStart + gSynOffset + ky * sy;
            CL_MEM_GLOBAL float * weightY = &(weights[wIdx + ky*syw]);
            //Summing into post buffer indexed by localIndex
            pvpatch_accumulate(nk, gSynY, a, weightY, (void*)0);
         }
      }
      //barrier(CLK_LOCAL_MEM_FENCE);
      //
      ////Copy post buf to gsyn start
      //CL_MEM_GLOBAL float* gSynStart = &(postGSyn[kPostRes]);
      //for(int ky = 0; ky < groupYSize; ky++){
      //   CL_MEM_GLOBAL float* globalGSynY = gSynStart + ky * sy;
      //   CL_MEM_LOCAL float* localGSynY = localGSynStart + ky * postBufYStride;
      //   for(int kxf = 0; kxf < postBufNumXF; kxf++){
      //      globalGSynY[kxf] = localGSynY[kxf];
      //   }
      //}
#endif
}
