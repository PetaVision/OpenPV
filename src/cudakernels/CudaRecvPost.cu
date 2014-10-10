#include "CudaRecvPost.hpp"
#include "../arch/cuda/cuda_util.hpp"
#include "conversions.hcu"

namespace PVCuda{

#ifdef PV_USE_CUDNN
#include <cudnn.h>

//Function to permutate ordering from [outFeature, ny, nx, inFeature] to [outFeature, inFeature, ny, nx]
//Coallessed in input
__global__
void CudaPermutePVToCudnn(float* dest, float* src, int outFeatures, int ny, int nx, int inFeatures){
   //kSrc into pre
   int kSrc = (blockIdx.x * blockDim.x) + threadIdx.x;
   if(kSrc < outFeatures * ny * nx * inFeatures){
      int kOF = kSrc/(ny*nx*inFeatures);
      int kY  = (kSrc % (ny*nx*inFeatures))/(nx*inFeatures);
      int kX  = (kSrc % (nx*inFeatures))/inFeatures;
      int kIF = (kSrc % inFeatures);

      int sOF = inFeatures * ny * nx;
      int sIF = ny * nx;
      int sY  = nx;

      int kDest = kOF * sOF + kIF * sIF + kY * sY + kX;

      //if(kSrc == 105344){
      //   printf("break here\n");
      //}

      dest[kDest] = src[kSrc];
   }
}

//Weights need to be reversed for cudnn
__global__
void CudaPermuteWeightsPVToCudnn(float* dest, float* src, int outFeatures, int ny, int nx, int inFeatures){
   //kSrc into pre
   int kSrc = (blockIdx.x * blockDim.x) + threadIdx.x;
   if(kSrc < outFeatures * ny * nx * inFeatures){
      int kOF = kSrc/(ny*nx*inFeatures);
      int kY  = (kSrc % (ny*nx*inFeatures))/(nx*inFeatures);
      int kX  = (kSrc % (nx*inFeatures))/inFeatures;
      int kIF = (kSrc % inFeatures);

      int sOF = inFeatures * ny * nx;
      int sIF = ny * nx;
      int sY  = nx;

      int kDest = kOF * sOF + kIF * sIF + (ny-kY-1) * sY + (nx-kX-1);

      //if(kSrc == 105344){
      //   printf("break here\n");
      //}

      dest[kDest] = src[kSrc];
   }
}

__global__
void CudaPermuteCudnnToPV(float* dest, float* src, int outFeatures, int ny, int nx, int inFeatures){
   //kSrc into pre
   int kSrc = (blockIdx.x * blockDim.x) + threadIdx.x;
   if(kSrc < outFeatures * ny * nx * inFeatures){

      int kOF = kSrc/(inFeatures*ny*nx);
      int kIF = (kSrc % (inFeatures*ny*nx))/(ny*nx);
      int kY  = (kSrc % (ny*nx))/nx;
      int kX  = (kSrc % (nx));

      int sOF = ny * nx * inFeatures;
      int sY  = nx * inFeatures;
      int sX  = inFeatures;

      int kDest = kOF * sOF + kY * sY + kX * sX + kIF;

      //if(kSrc == 315978){
      //   printf("break here\n");
      //}
      
      //Copy
      dest[kDest] = src[kSrc];
   }
}

__global__
void testEquality(float* eq1, float* eq2, int outFeatures, int ny, int nx, int inFeatures){
   //kSrc into pre
   int kSrc = (blockIdx.x * blockDim.x) + threadIdx.x;
   if(kSrc < outFeatures * ny * nx * inFeatures){
      if(fabs(eq1[kSrc] - eq2[kSrc]) > 1e-10){
         printf("Broken here\n");
      }
   }
}
#endif

//Kernel code
__global__
void HyPerLayer_recv_post(recv_post_params params){
   ////Shared memory buffers are declared
   extern __shared__ char sharedMem[];
   __shared__ float* preBuffer;
   __shared__ float* postBuffer;
   __shared__ float* weightsBuffer;

   postBuffer = (float*)sharedMem;
   weightsBuffer = (float*)(&(postBuffer[params.postBufNum]));

   if(params.preDataLocal){
      preBuffer = (float*)(&(weightsBuffer[params.weightsBufNum]));
   }

   //Ordered this way because threads vary fastest in x, then y, then z
   //Mapped to petavision order of f, x, and y

   int localF = blockDim.x;
   int localX = blockDim.y;
   int localY = blockDim.z;
   
   int localFIndex = threadIdx.x;
   int localXIndex = threadIdx.y;
   int localYIndex = threadIdx.z;

   int fTargetRes = (blockIdx.x * blockDim.x) + threadIdx.x;
   int xTargetRes = (blockIdx.y * blockDim.y) + threadIdx.y;
   int yTargetRes = (blockIdx.z * blockDim.z) + threadIdx.z;

   ////Calculate kTargetRes based on x, y, and f
   int kTargetRes = kIndex(xTargetRes, yTargetRes, fTargetRes, params.nxRes, params.nyRes, params.nf);

   int kTargetExt = kIndexExtended(kTargetRes, params.nxRes, params.nyRes, params.nf, params.nblt, params.nbrt, params.nbdn, params.nbup);

   //Each wIdx should be shared since each workgroup convolves one weight kernel
   __shared__ int wIdx;
   if(localXIndex == 0 && localYIndex == 0){
      //Change restricted to extended post neuron
      int kernelIndex;
      if(params.sharedWeights == 1){
         kernelIndex = params.patch2datalookuptable[kTargetExt];
      }
      else{
         kernelIndex = kTargetExt;
      }
      wIdx = kernelIndex * params.nxp * params.nyp * params.nfp;
   }

   //Get top left most neuron in the group
   __shared__ long localStartSourceExt;
   long startSourceExt;
   if(params.preDataLocal){
      if(localXIndex == 0 && localYIndex == 0 && localFIndex == 0){
         localStartSourceExt = params.startSourceExtBuf[kTargetRes];
      }
   }
   else{
      startSourceExt = params.startSourceExtBuf[kTargetRes];
   }

   int localIndex = kIndex(localXIndex, localYIndex, localFIndex, localX, localY, localF);

   postBuffer[localIndex] = 0;
      
   int numXfBuffer = params.localBufSizeX * params.nfp;
   int numWeightsBuffer = params.nxp * params.nfp;

   int xOffset = localXIndex * params.preToPostScaleX;
   //int yOffset = localYIndex * params.preToPostScaleY;

   int numCopyThreads = localF * localX * localY < warpSize ? localF * localX * localY : warpSize;
   
   //Wait for shared memory loads
   __syncthreads();

   for(int ky = 0; ky < params.nyp; ky++){
      //Copy global to local, do this with all threads
      if(params.preDataLocal){
         //Pre buffer
         if(localIndex < numCopyThreads){
            for(int i = localIndex; i < numXfBuffer; i+= numCopyThreads){
               preBuffer[i] = params.preData[localStartSourceExt + ky * params.sy + i];
            }
         }
      }

      //Weights
      if(localIndex < numCopyThreads){
         for(int i = localIndex; i < numWeightsBuffer; i+= numCopyThreads){
            weightsBuffer[i] = params.weights[wIdx + ky * params.syp + i];
         }
      }
      //The actual pre buffer index
      __syncthreads();

      float* activityY;
      if(params.preDataLocal){
         activityY = &(preBuffer[xOffset * params.nfp]);
      }
      else{
         activityY = &(params.preData[startSourceExt + ky * params.sy]);
      }

      float* weightY = weightsBuffer;
      //float* weightY = &(params.weights[wIdx + ky * params.syp]);

      //Summing into post buffer indexed by localIndex
      int k;
      for (k = 0; k < params.numPerStride; k++) {
         postBuffer[localIndex] += activityY[k]*weightY[k]*params.dt_factor;
      }
      __syncthreads();
   }
   ////Sum into global memory
   params.postGsyn[kTargetRes] += postBuffer[localIndex];
   //if(isnan(params.postGsyn[kTargetRes])){
   //   printf("breakHere\n");
   //}
}


CudaRecvPost::CudaRecvPost(CudaDevice* inDevice):CudaKernel(inDevice){
}

CudaRecvPost::~CudaRecvPost(){
#ifdef PV_USE_CUDNN
   if(params.v_inputDescriptor){
      cudnnTensor4dDescriptor_t inputDescriptor = (cudnnTensor4dDescriptor_t) params.v_inputDescriptor;
      cudnnDestroyTensor4dDescriptor(inputDescriptor);
   }
   if(params.v_filterDescriptor){
      cudnnFilterDescriptor_t filterDescriptor = (cudnnFilterDescriptor_t) params.v_filterDescriptor;
      cudnnDestroyFilterDescriptor(filterDescriptor);
   }
   if(params.v_outputDescriptor){
      cudnnTensor4dDescriptor_t outputDescriptor = (cudnnTensor4dDescriptor_t) params.v_outputDescriptor;
      cudnnDestroyTensor4dDescriptor(outputDescriptor);
   }
   if(params.v_convDescriptor){
      cudnnConvolutionDescriptor_t convDescriptor = (cudnnConvolutionDescriptor_t) params.v_convDescriptor;
      cudnnDestroyConvolutionDescriptor(convDescriptor);
   }
#endif
}

void CudaRecvPost::setArgs(
      const int nxRes, //num post neurons
      const int nyRes,
      const int nf,

      const int nblt, //Border of orig
      const int nbrt, //Border of orig
      const int nbdn, //Border of orig
      const int nbup, //Border of orig

      const int preNx,
      const int preNy,
      const int preNf,
      const int preNblt,
      const int preNbrt,
      const int preNbup,
      const int preNbdn,

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

      /* long* */  CudaBuffer* startSourceExtBuf,
      /* float* */ CudaBuffer* preData,
      /* float* */ CudaBuffer* weights,
      /* float* */ CudaBuffer* postGsyn,
#ifdef PV_USE_CUDNN
      /* float* */ CudaBuffer* cudnn_preData,
      /* float* */ CudaBuffer* cudnn_weights,
      /* float* */ CudaBuffer* cudnn_gSyn,
#endif
      /* int* */   CudaBuffer* patch2datalookuptable,

      const bool preDataLocal
   ){
   params.nxRes = nxRes;
   params.nyRes = nyRes;
   params.nf = nf;

   params.nblt = nblt;
   params.nbrt = nbrt;
   params.nbdn = nbdn;
   params.nbup = nbup;

   params.preNx   = preNx;
   params.preNy   = preNy;
   params.preNf   = preNf;
   params.preNblt = preNblt;
   params.preNbrt = preNbrt;
   params.preNbup = preNbup;
   params.preNbdn = preNbdn;

   params.nxp = nxp;
   params.nyp = nyp;
   params.nfp = nfp;

   params.localBufSizeX = localBufSizeX;
   params.localBufSizeY = localBufSizeY;
   params.preToPostScaleX = preToPostScaleX;
   params.preToPostScaleY = preToPostScaleY;

   params.sy = sy;
   params.syp = syp;
   params.numPerStride = numPerStride;
   params.dt_factor = dt_factor;
   params.sharedWeights = sharedWeights;

   params.startSourceExtBuf = (long*)startSourceExtBuf->getPointer();
   params.preData = (float*)preData->getPointer();
   params.weights = (float*)weights->getPointer();
   params.postGsyn = (float*)postGsyn->getPointer();
#ifdef PV_USE_CUDNN
   params.cudnn_weights = (float*)cudnn_weights->getPointer();
   params.cudnn_preData = (float*)cudnn_preData->getPointer();
   params.cudnn_gSyn = (float*)cudnn_gSyn->getPointer();
#endif
   params.patch2datalookuptable = (int*)patch2datalookuptable->getPointer();

   params.warpSize = device->get_warp_size();

   params.preDataLocal = preDataLocal;

#ifdef PV_USE_CUDNN
   //Can only do many pre to one post
   if(preToPostScaleX < 1 || preToPostScaleY < 1){
      printf("One to Many case not implemented with CUDNN\n");
      exit(-1);
   }

   //Set up pre descriptor
   cudnnTensor4dDescriptor_t inputDescriptor;
   cudnnStatus_t status = cudnnCreateTensor4dDescriptor(&inputDescriptor);
   assert(status == CUDNN_STATUS_SUCCESS);
   status = cudnnSetTensor4dDescriptor(inputDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
      1, //Number of images
      params.preNf, //Number of feature maps per image
      params.preNy + params.preNbup + params.preNbdn, //Height of each feature map
      params.preNx + params.preNblt + params.preNbrt); //Width of each feature map
   if(status != CUDNN_STATUS_SUCCESS){
      switch(status){
         case CUDNN_STATUS_BAD_PARAM:
            printf("cuDNN bad parameter\n");
            break;
         default:
            printf("cuDNN unknown error code %d\n", status);
      }
      exit(-1);
   }
   assert(status == CUDNN_STATUS_SUCCESS);
   params.v_inputDescriptor = (void*)inputDescriptor;

   //Set up filter descriptor
   cudnnFilterDescriptor_t filterDescriptor;
   status = cudnnCreateFilterDescriptor(&filterDescriptor);
   assert(status == CUDNN_STATUS_SUCCESS);
   status = cudnnSetFilterDescriptor(filterDescriptor, CUDNN_DATA_FLOAT,
      params.nf, //Number of output feature maps
      params.nfp, //Number of input feature maps
      params.nyp, //Height of each filter
      params.nxp); //Width of each filter
   assert(status == CUDNN_STATUS_SUCCESS);
   params.v_filterDescriptor = (void*)filterDescriptor;

   //There's the case where the border of pre is made bigger through other connections. Need to calculate difference
   //between current recv border and actual recv border
   int actualXBorder = (params.nxp-params.preToPostScaleX)/2;
   int actualYBorder = (params.nyp-params.preToPostScaleY)/2;

   assert(params.preNblt == params.preNbrt);
   assert(params.preNbup == params.preNbdn);

   int diffX = actualXBorder - params.preNblt;
   int diffY = actualYBorder - params.preNbup;

   assert(diffX <= 0 && diffY <= 0);

   //Set convolution descriptor
   cudnnConvolutionDescriptor_t convDescriptor;
   status = cudnnCreateConvolutionDescriptor(&convDescriptor);
   assert(status == CUDNN_STATUS_SUCCESS);
   status = cudnnSetConvolutionDescriptor(convDescriptor, inputDescriptor, filterDescriptor,
      //params.nyp-params.preToPostScaleY-1,
      //params.nxp-params.preToPostScaleX-1,  //zero-padding height and width
      diffY,
      diffX,  //zero-padding height and width
      params.preToPostScaleY, //Vertical filter stride
      params.preToPostScaleX, //Horizontal filter stride
      1, 1, //upscale the input in x/y direction
      CUDNN_CONVOLUTION);
   assert(status == CUDNN_STATUS_SUCCESS);
   params.v_convDescriptor = (void*)convDescriptor;

   //Query output layout and check with PV layout
   int out_n, out_c, out_h, out_w;
   status = cudnnGetOutputTensor4dDim(convDescriptor, CUDNN_CONVOLUTION_FWD,
      &out_n, //num images
      &out_c, //num output features
      &out_h, //output height
      &out_w); //output width
   assert(status == CUDNN_STATUS_SUCCESS);

   //Make sure dimensions match up with PV layer
   if(out_n != 1 || out_h != nyRes || out_w != nxRes || out_c != nf){
      printf("CUDNN:: Dimensions don't match: \n");
      printf("Dimensions of output tensor (n, y, x, f): %d, %d, %d, %d\n", out_n, out_h, out_w, out_c);
      printf("Dimensions of output PV layer (n, y, x, f): 1, %d, %d, %d\n", nyRes, nxRes, nf);
      exit(-1);
   }

   //Set up output descriptor
   cudnnTensor4dDescriptor_t outputDescriptor;
   status = cudnnCreateTensor4dDescriptor(&outputDescriptor);
   assert(status == CUDNN_STATUS_SUCCESS);
   status = cudnnSetTensor4dDescriptor(outputDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
      1, //Number of images
      nf, //Number of feature maps per image
      nyRes, //ny restricted
      nxRes); //nx restricted
   if(status != CUDNN_STATUS_SUCCESS){
      switch(status){
         case CUDNN_STATUS_BAD_PARAM:
            printf("cuDNN bad parameter\n");
            break;
         default:
            printf("cuDNN unknown error code %d\n", status);
      }
      exit(-1);
   }
   assert(status == CUDNN_STATUS_SUCCESS);
   params.v_outputDescriptor = (void*)outputDescriptor;
#endif

   setArgsFlag();
}

#ifdef PV_USE_CUDNN
void CudaRecvPost::permuteDatastorePVToCudnn(){
   //Ext pre activity
   int ny = params.preNy + params.preNbup + params.preNbdn;
   int nx = params.preNx + params.preNblt + params.preNbrt;
   int nf = params.preNf;

   //Calculate grid and work size
   int numNeurons = ny * nx * nf;
   int blockSize = device->get_max_threads();
   //Ceil to get all weights
   int gridSize = ceil((float)numNeurons/blockSize);
   //printf("Calling activity PVToCudnn with (%d, %d, %d)\n", ny, nx, nf);

   device->syncDevice();

   //Call function
   CudaPermutePVToCudnn<<<gridSize, blockSize, 0, device->getStream()>>>(params.cudnn_preData, params.preData, 1, ny, nx, nf);
   handleCallError();

   device->syncDevice();
}

void CudaRecvPost::permuteGSynPVToCudnn(int channel){
   //Res post activity
   int ny = params.nyRes;
   int nx = params.nxRes;
   int nf = params.nf;

   //Calculate grid and work size
   int numNeurons = ny * nx * nf;
   float* gSynPatchHead = &(params.postGsyn[numNeurons * channel]);

   int blockSize = device->get_max_threads();
   //Ceil to get all weights
   int gridSize = ceil((float)numNeurons/blockSize);
   //Call function
   //printf("Calling gsyn PVToCudnn with (%d, %d, %d)\n", ny, nx, nf);
   CudaPermutePVToCudnn<<<gridSize, blockSize, 0, device->getStream()>>>(params.cudnn_gSyn, gSynPatchHead, 1, ny, nx, nf);
   handleCallError();
}

void CudaRecvPost::permuteGSynCudnnToPV(int channel){
   //Res post activity
   int ny = params.nyRes;
   int nx = params.nxRes;
   int nf = params.nf;

   //Calculate grid and work size
   int numNeurons = ny * nx * nf;
   float* gSynPatchHead = &(params.postGsyn[numNeurons * channel]);

   int blockSize = device->get_max_threads();
   //Ceil to get all weights
   int gridSize = ceil((float)numNeurons/blockSize);
   //Call function
   //printf("Calling gsyn Cudnn To PV with (%d, %d, %d)\n", ny, nx, nf);
   CudaPermuteCudnnToPV<<<gridSize, blockSize, 0, device->getStream()>>>(gSynPatchHead, params.cudnn_gSyn, 1, ny, nx, nf);
   handleCallError();
}

void CudaRecvPost::permuteWeightsPVToCudnn(){
   //outFeatures is number of kernels
   //Note this is only the case for many to one and one to one
   int outFeatures = params.nf;
   //Rest is patch sizes
   int ny = params.nyp;
   int nx = params.nxp;
   int inFeatures = params.nfp;

   //Calculate grid and work size
   int numWeights = outFeatures * ny * nx * inFeatures;
   int blockSize = device->get_max_threads();
   //Ceil to get all weights
   int gridSize = ceil((float)numWeights/blockSize);
   //Call function
   //printf("Calling weights PV To Cudnn with (%d, %d, %d, %d)\n", outFeatures, ny, nx, inFeatures);
   CudaPermuteWeightsPVToCudnn<<<gridSize, blockSize, 0, device->getStream()>>>(params.cudnn_weights, params.weights, outFeatures, ny, nx, inFeatures);
   handleCallError();
}

#endif

int CudaRecvPost::do_run(){
   
#ifdef PV_USE_CUDNN
   cudnnHandle_t handle = (cudnnHandle_t) device->getCudnnHandle();
   cudnnTensor4dDescriptor_t inputDescriptor = (cudnnTensor4dDescriptor_t) params.v_inputDescriptor;
   cudnnFilterDescriptor_t filterDescriptor = (cudnnFilterDescriptor_t) params.v_filterDescriptor;
   cudnnTensor4dDescriptor_t outputDescriptor = (cudnnTensor4dDescriptor_t) params.v_outputDescriptor;
   cudnnConvolutionDescriptor_t convDescriptor = (cudnnConvolutionDescriptor_t) params.v_convDescriptor;

   int status = cudnnConvolutionForward(handle,
      inputDescriptor, params.cudnn_preData,
      filterDescriptor, params.cudnn_weights,
      convDescriptor,
      outputDescriptor, params.cudnn_gSyn,
      CUDNN_RESULT_ACCUMULATE);

   assert(status == CUDNN_STATUS_SUCCESS);
   
#else
   params.postBufNum = block_size.x * block_size.y * block_size.z;

   if(params.preDataLocal){
      params.preBufNum = params.localBufSizeX * params.nfp;
   }
   else{
      params.preBufNum = 0;
   }

   params.weightsBufNum = params.nxp * params.nfp;

   size_t sharedSize = sizeof(float) * (params.preBufNum + params.postBufNum + params.weightsBufNum);

   if(sharedSize > device->get_local_mem()){
      printf("gpu post run: given shared memory size of %zu is bigger than allowed shared memory size of %zu\n", sharedSize, device->get_local_mem());
      exit(-1);
   }

   if(block_size.x != 1){
      printf("gpu post run: numFLocal must be 1\n");
      exit(-1);
   }
   
   if(params.preDataLocal){
      if(block_size.z != 1){
         printf("gpu post run: numYLocal must be 1 if using local pre data\n");
         exit(-1);
      }
   }

   HyPerLayer_recv_post<<<grid_size, block_size, sharedSize, device->getStream()>>>(params);
   handleCallError();
#endif

   return 0;
}

}
