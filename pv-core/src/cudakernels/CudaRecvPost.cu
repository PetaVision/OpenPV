#include "CudaRecvPost.hpp"
#include "../arch/cuda/cuda_util.hpp"
#include "conversions.hcu"

namespace PVCuda{

#ifdef PV_USE_CUDNN
#include <cudnn.h>

//Function to change PV representation to CUDNN reprsentation
//Does 2 things: permutate ordering from [outFeature, ny, nx, inFeature] to [outFeature, inFeature, ny, nx]
//Reshapes the matrix if manyScale > 1 to map different "many" kernels into feature dimension
//Coallessed in input
__global__
void CudaPermutePVToCudnn(float* dest, float* src, int outFeatures, int ny, int nx, int inFeatures, int manyScaleX, int manyScaleY, int cropX, int cropY){
   //parameter dimensions are in source PV format
   int destNx = (nx-2*cropX)/manyScaleX;
   int destNy = (ny-2*cropY)/manyScaleY;
   int destInFeatures = inFeatures*manyScaleX*manyScaleY;

   int kSrc = (blockIdx.x * blockDim.x) + threadIdx.x;
   if(kSrc < outFeatures * ny * nx * inFeatures){
      int kOF = kSrc/(ny*nx*inFeatures);
      int kY  = (kSrc % (ny*nx*inFeatures))/(nx*inFeatures);
      int kX  = (kSrc % (nx*inFeatures))/inFeatures;
      int kIF = (kSrc % inFeatures);

      //check if in bounds
      if(kX < cropX || kX >= nx-cropX){ 
         return;
      }
      else{
         kX = kX - cropX;
      }
      if(kY < cropY || kY >= ny-cropY){
         return;
      }
      else{
         kY = kY - cropY;
      }

      //Recalculate x, y, and f based on manyScale
      kIF = kIF + inFeatures * (kX % manyScaleX + (kY % manyScaleY) * manyScaleX);
      kX = kX/manyScaleX;
      kY = kY/manyScaleY;

      int sOF = destInFeatures * destNy * destNx;
      int sIF = destNy * destNx;
      int sY  = destNx;

      int kDest = kOF * sOF + kIF * sIF + kY * sY + kX;

      dest[kDest] = src[kSrc];
   }
}

//Weights need to be reversed for cudnn
//No need to account for many because the PV representation matches with how gsyn was reshaped.
__global__
void CudaPermuteWeightsPVToCudnn(float* dest, float* src, int outFeatures, int ny, int nx, int inFeatures, int manyScaleX, int manyScaleY){
   //Parameter dimensions are PV source dimensions
   int kSrc = (blockIdx.x * blockDim.x) + threadIdx.x;
   if(kSrc < outFeatures * manyScaleX * manyScaleY * ny * nx * inFeatures){
      int kOF = kSrc/(ny*nx*inFeatures);
      int kY  = (kSrc % (ny*nx*inFeatures))/(nx*inFeatures);
      int kX  = (kSrc % (nx*inFeatures))/inFeatures;
      int kIF = (kSrc % inFeatures);

      int sOF = inFeatures * ny * nx;
      int sIF = ny * nx;
      int sY  = nx;

      int kDest = kOF * sOF + kIF * sIF + (ny-kY-1) * sY + (nx-kX-1);

      dest[kDest] = src[kSrc];
   }
}

__global__
void CudaPermuteCudnnToPV(float* dest, float* src, int outFeatures, int ny, int nx, int inFeatures, int manyScaleX, int manyScaleY){
   //parameter dimensions are in dest PV format
   int srcNx = nx/manyScaleX;
   int srcNy = ny/manyScaleY;
   int srcInFeatures = inFeatures*manyScaleX*manyScaleY;

   int kDest = (blockIdx.x * blockDim.x) + threadIdx.x;
   if(kDest < outFeatures * ny * nx * inFeatures){
      int kOF = kDest/(ny*nx*inFeatures);
      int kY  = (kDest % (ny*nx*inFeatures))/(nx*inFeatures);
      int kX  = (kDest % (nx*inFeatures))/inFeatures;
      int kIF = (kDest % inFeatures);

      //Recalculate x, y, and f based on manyScale
      kIF = kIF + inFeatures * (kX % manyScaleX + (kY % manyScaleY) * manyScaleX);
      kX = kX/manyScaleX;
      kY = kY/manyScaleY;

      int sOF = srcInFeatures * srcNy * srcNx;
      int sIF = srcNy * srcNx;
      int sY  = srcNx;

      int kSrc = kOF * sOF + kIF * sIF + kY * sY + kX;

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

void cudnnHandleError(cudnnStatus_t status, const char* errStr){
   if(status != CUDNN_STATUS_SUCCESS){
      printf("CUDNN %s error: %s", errStr, cudnnGetErrorString(status));
      exit(-1);
      return;
   }
   return;
}

#endif

//Kernel code
__global__
void HyPerLayer_recv_post(recv_post_params params, int batch){
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

   int preBatchOffset = batch * (params.preNx + params.preNblt + params.preNbrt) * (params.preNy + params.preNbup + params.preNbdn) * params.preNf; 

   for(int ky = 0; ky < params.nyp; ky++){
      //Copy global to local, do this with all threads
      if(params.preDataLocal){
         //Pre buffer
         if(localIndex < numCopyThreads){
            for(int i = localIndex; i < numXfBuffer; i+= numCopyThreads){
               preBuffer[i] = params.preData[preBatchOffset + localStartSourceExt + ky * params.sy + i];
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
         activityY = &(params.preData[preBatchOffset + startSourceExt + ky * params.sy]);
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
   int postBatchOffset = batch * params.nxRes * params.nyRes * params.nf; 
   params.postGsyn[postBatchOffset + kTargetRes] += postBuffer[localIndex];
}


CudaRecvPost::CudaRecvPost(CudaDevice* inDevice):CudaKernel(inDevice){
   //inDevice->incrementConvKernels();
}

CudaRecvPost::~CudaRecvPost(){
#ifdef PV_USE_CUDNN
   if(params.v_inputDescriptor){
      cudnnTensorDescriptor_t inputDescriptor = (cudnnTensorDescriptor_t) params.v_inputDescriptor;
      cudnnDestroyTensorDescriptor(inputDescriptor);
   }
   if(params.v_filterDescriptor){
      cudnnFilterDescriptor_t filterDescriptor = (cudnnFilterDescriptor_t) params.v_filterDescriptor;
      cudnnDestroyFilterDescriptor(filterDescriptor);
   }
   if(params.v_outputDescriptor){
      cudnnTensorDescriptor_t outputDescriptor = (cudnnTensorDescriptor_t) params.v_outputDescriptor;
      cudnnDestroyTensorDescriptor(outputDescriptor);
   }
   if(params.v_convDescriptor){
      cudnnConvolutionDescriptor_t convDescriptor = (cudnnConvolutionDescriptor_t) params.v_convDescriptor;
      cudnnDestroyConvolutionDescriptor(convDescriptor);
   }
   if(params.v_convAlgo){
      cudnnConvolutionFwdAlgo_t* convAlgo = (cudnnConvolutionFwdAlgo_t*) params.v_convAlgo;
      delete convAlgo;
   }
   if(params.cudnn_workspace){
      handleError(cudaFree(params.cudnn_workspace), "Freeing workspace pointer");
   }
   if(params.workspaceSize){
      delete params.workspaceSize;
   }
#endif
}

void CudaRecvPost::setArgs(
      const int nbatch,
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
   params.nbatch = nbatch;
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
   //CUDNN code
   //Calculate how much space is left on the gpu for the workspace memory
   //Do not add to device's count since there might be more than one kernel that needs workspace memory
   size_t workspaceMem = device->getMemory()/device->getNumConvKernels();

   int strideX, strideY;
   int actualXBorder, actualYBorder;
   assert(params.preNblt == params.preNbrt);
   assert(params.preNbup == params.preNbdn);
   //One to many case
   if(preToPostScaleX < 1){
      float fmanyScale = (float)1/params.preToPostScaleX;
      //Make sure manyScale is an actual integer
      assert(ceilf(fmanyScale) == fmanyScale);
      params.manyScaleX = fmanyScale;
      fmanyScale = (float)1/params.preToPostScaleY;
      assert(ceilf(fmanyScale) == fmanyScale);
      params.manyScaleY = fmanyScale;
      strideX = 1;
      strideY = 1;

      //Patch sizes must be odd multiple of many
      if(nxp % 2 == 0 || nyp % 2 == 0){
         printf("cuDNN: Running on a one to many connection with CUDNN must have patch size (%d, %d) be an odd muliple of many (%d, %d)\n", nxp*params.manyScaleX, nyp*params.manyScaleY, params.manyScaleX, params.manyScaleY);
         exit(-1);
      }

      
      //There's the case where the border of pre is made bigger through other connections. Need to calculate difference
      //between current recv border and actual recv border
      //This is calculating what the border would be if this was a one to one connection
      actualXBorder = params.nxp/2;
      actualYBorder = params.nyp/2;
   }
   //Many to one or one to one case
   else{
      params.manyScaleX = 1;
      params.manyScaleY = 1;
      assert(ceilf(preToPostScaleX) == preToPostScaleX);
      assert(ceilf(preToPostScaleY) == preToPostScaleY);
      strideX = preToPostScaleX;
      strideY = preToPostScaleY;

      //There's the case where the border of pre is made bigger through other connections. Need to calculate difference
      //between current recv border and actual recv border
      actualXBorder = (params.nxp-params.preToPostScaleX)/2;
      actualYBorder = (params.nyp-params.preToPostScaleY)/2;
   }

   //params.diffX = actualXBorder - params.preNblt;
   //params.diffY = actualYBorder - params.preNbup;

   //diffX is positive value of cropping
   params.diffX = params.preNblt - actualXBorder;
   params.diffY = params.preNbup - actualYBorder;

   //assert(diffX <= 0 && diffY <= 0);

   //Set up pre descriptor
   cudnnTensorDescriptor_t inputDescriptor;
   cudnnStatus_t status = cudnnCreateTensorDescriptor(&inputDescriptor);
   cudnnHandleError(status, "Create input tensor descriptor");

   status = cudnnSetTensor4dDescriptor(inputDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
      nbatch, //Number of images
      params.preNf, //Number of feature maps per image
      params.preNy + params.preNbup + params.preNbdn - 2*params.diffY, //Height of each feature map
      params.preNx + params.preNblt + params.preNbrt - 2*params.diffX); //Width of each feature map
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
   cudnnHandleError(status, "Set input tensor descriptor");
   params.v_inputDescriptor = (void*)inputDescriptor;

   //Set up filter descriptor
   cudnnFilterDescriptor_t filterDescriptor;
   status = cudnnCreateFilterDescriptor(&filterDescriptor);
   cudnnHandleError(status, "Create filter tensor descriptor");
   status = cudnnSetFilter4dDescriptor(filterDescriptor, CUDNN_DATA_FLOAT,
      params.nf * params.manyScaleX * params.manyScaleY, //Number of output feature maps. For one to many, output feature maps are repeated for each kernel
      params.nfp, //Number of input feature maps
      params.nyp, //Height of each filter
      params.nxp); //Width of each filter
   cudnnHandleError(status, "Set filter tensor descriptor");
   params.v_filterDescriptor = (void*)filterDescriptor;

   //Set convolution descriptor
   cudnnConvolutionDescriptor_t convDescriptor;
   status = cudnnCreateConvolutionDescriptor(&convDescriptor);
   cudnnHandleError(status, "Create convolution tensor descriptor");
   status = cudnnSetConvolution2dDescriptor(convDescriptor,
      //params.nyp-params.preToPostScaleY-1,
      //params.nxp-params.preToPostScaleX-1,  //zero-padding height and width
      0,
      0,  //zero-padding height and width
      strideY, //Vertical filter stride
      strideX, //Horizontal filter stride
      1, 1, //upscale the input in x/y direction
      CUDNN_CONVOLUTION);
   cudnnHandleError(status, "Set convolution tensor descriptor");
   params.v_convDescriptor = (void*)convDescriptor;

   //Query output layout and check with PV layout
   int out_n, out_c, out_h, out_w;
   status = cudnnGetConvolution2dForwardOutputDim(convDescriptor, inputDescriptor, filterDescriptor,
      &out_n, //num images
      &out_c, //num output features
      &out_h, //output height
      &out_w); //output width
   cudnnHandleError(status, "Get output tensor descriptor");

   //Make sure dimensions match up with PV layer
   if(out_n != nbatch || out_h != nyRes/params.manyScaleY || out_w != nxRes/params.manyScaleX || out_c != nf*params.manyScaleX*params.manyScaleY){
      printf("CUDNN:: Dimensions don't match: \n");
      printf("Dimensions of output tensor (n, y, x, f): %d, %d, %d, %d\n", out_n, out_h, out_w, out_c);
      printf("Scaled dimensions of output PV layer (n, y, x, f): %d, %d, %d, %d\n", nbatch, nyRes/params.manyScaleY, nxRes/params.manyScaleX, nf*params.manyScaleX*params.manyScaleY);
      printf("Actual dimensions of output PV layer (n, y, x, f): %d, %d, %d, %d\n", nbatch, nyRes, nxRes, nf);
      exit(-1);
   }

   //Set up output descriptor
   cudnnTensorDescriptor_t outputDescriptor;
   status = cudnnCreateTensorDescriptor(&outputDescriptor);
   cudnnHandleError(status, "Create output tensor descriptor");
   status = cudnnSetTensor4dDescriptor(outputDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
      nbatch, //Number of images
      nf*params.manyScaleX*params.manyScaleY, //Number of feature maps per image
      nyRes/params.manyScaleY, //ny restricted
      nxRes/params.manyScaleX); //nx restricted
   cudnnHandleError(status, "Set output tensor descriptor");
   params.v_outputDescriptor = (void*)outputDescriptor;

   //Calculate and set up best forward conv algorithm to use
   cudnnHandle_t handle = (cudnnHandle_t) device->getCudnnHandle();
   cudnnConvolutionFwdAlgo_t* convAlgo = new cudnnConvolutionFwdAlgo_t();

   status = cudnnGetConvolutionForwardAlgorithm(
      handle,
      inputDescriptor,
      filterDescriptor,
      convDescriptor,
      outputDescriptor,
      //TODO: use this flag, but we need to calculate how much free space is left on the GPU and pass it in as next argument
      CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
      //CUDNN_CONVOLUTION_FWD_NO_WORKSPACE,
      workspaceMem,
      convAlgo
   );
   cudnnHandleError(status, "Get convolution forward algorithm");
   params.v_convAlgo = (void*) convAlgo;

   //Based on algortihm, allocate workspace memory for GPU
   size_t* temp = new size_t();
   status = cudnnGetConvolutionForwardWorkspaceSize(
      handle,
      inputDescriptor,
      filterDescriptor,
      convDescriptor,
      outputDescriptor,
      *convAlgo,
      temp
   );
   params.workspaceSize = temp;
   cudnnHandleError(status, "Get convolution forward workspace size");

   //Allocate workspace based on size
   handleError(cudaMalloc(&params.cudnn_workspace, *params.workspaceSize), "Cudnn workspace cudaMalloc");

#endif

   setArgsFlag();
}

#ifdef PV_USE_CUDNN
void CudaRecvPost::permuteDatastorePVToCudnn(){
   //Ext pre activity
   int ny = params.preNy + params.preNbup + params.preNbdn;
   int nx = params.preNx + params.preNblt + params.preNbrt;
   int nf = params.preNf;
   int nbatch = params.nbatch;

   //Calculate grid and work size
   int numNeurons = nbatch * ny * nx * nf;
   int blockSize = device->get_max_threads();
   //Ceil to get all weights
   int gridSize = ceil((float)numNeurons/blockSize);
   //printf("Calling activity PVToCudnn with (%d, %d, %d)\n", ny, nx, nf);

   device->syncDevice();

   //Call function
   //Datastore will never get reshaped, so manyScale will always be 1
   CudaPermutePVToCudnn<<<gridSize, blockSize, 0, device->getStream()>>>(params.cudnn_preData, params.preData, nbatch, ny, nx, nf, 1, 1, params.diffX, params.diffY);
   handleCallError("Permute PV to CUDNN");
}

void CudaRecvPost::permuteGSynPVToCudnn(int channel){
   //Res post activity
   int ny = params.nyRes;
   int nx = params.nxRes;
   int nf = params.nf;
   int nbatch = params.nbatch;

   //Calculate grid and work size
   int numNeurons = nbatch * ny * nx * nf;
   float* gSynPatchHead = &(params.postGsyn[numNeurons * channel]);

   int blockSize = device->get_max_threads();
   //Ceil to get all weights
   int gridSize = ceil((float)numNeurons/blockSize);
   //Call function
   CudaPermutePVToCudnn<<<gridSize, blockSize, 0, device->getStream()>>>(params.cudnn_gSyn, gSynPatchHead, nbatch, ny, nx, nf, params.manyScaleX, params.manyScaleY, 0, 0);
   handleCallError("Permute GSyn PV to CUDNN");
}

void CudaRecvPost::permuteGSynCudnnToPV(int channel){
   //Res post activity
   int ny = params.nyRes;
   int nx = params.nxRes;
   int nf = params.nf;
   int nbatch = params.nbatch;

   //Calculate grid and work size
   int numNeurons = nbatch * ny * nx * nf;
   float* gSynPatchHead = &(params.postGsyn[numNeurons * channel]);

   int blockSize = device->get_max_threads();
   //Ceil to get all weights
   int gridSize = ceil((float)numNeurons/blockSize);
   //Call function
   //printf("Calling gsyn Cudnn To PV with (%d, %d, %d)\n", ny, nx, nf);
   CudaPermuteCudnnToPV<<<gridSize, blockSize, 0, device->getStream()>>>(gSynPatchHead, params.cudnn_gSyn, nbatch, ny, nx, nf, params.manyScaleX, params.manyScaleY);
   handleCallError("Permute GSyn CUDNN to PV");
}

//void CudaRecvPost::permuteWeightsPVToCudnn(){
//   //outFeatures is number of kernels
//   int outFeatures = params.nf;
//
//   //Rest is patch sizes
//   int ny = params.nyp;
//   int nx = params.nxp;
//   int inFeatures = params.nfp;
//
//   //Calculate grid and work size
//   int numWeights = outFeatures * params.manyScaleX * params.manyScaleY * ny * nx * inFeatures;
//   int blockSize = device->get_max_threads();
//   //Ceil to get all weights
//   int gridSize = ceil((float)numWeights/blockSize);
//   //Call function
//   //printf("Calling weights PV To Cudnn with (%d, %d, %d, %d)\n", outFeatures, ny, nx, inFeatures);
//   CudaPermuteWeightsPVToCudnn<<<gridSize, blockSize, 0, device->getStream()>>>(params.cudnn_weights, params.weights, outFeatures, ny, nx, inFeatures, params.manyScaleX, params.manyScaleY);
//   handleCallError("Permute weights PV to CUDNN");
//}

#endif

int CudaRecvPost::do_run(){
   
#ifdef PV_USE_CUDNN
   cudnnHandle_t handle = (cudnnHandle_t) device->getCudnnHandle();
   cudnnTensorDescriptor_t inputDescriptor = (cudnnTensorDescriptor_t) params.v_inputDescriptor;
   cudnnFilterDescriptor_t filterDescriptor = (cudnnFilterDescriptor_t) params.v_filterDescriptor;
   cudnnTensorDescriptor_t outputDescriptor = (cudnnTensorDescriptor_t) params.v_outputDescriptor;
   cudnnConvolutionDescriptor_t convDescriptor = (cudnnConvolutionDescriptor_t) params.v_convDescriptor;
   cudnnConvolutionFwdAlgo_t* convAlgo = (cudnnConvolutionFwdAlgo_t*) params.v_convAlgo;

   float scalingFactor = 1;

   cudnnStatus_t status = cudnnConvolutionForward(
      handle,
      &(scalingFactor),
      inputDescriptor,
      params.cudnn_preData,
      filterDescriptor,
      params.cudnn_weights,
      convDescriptor,
      *convAlgo,
      params.cudnn_workspace,
      *params.workspaceSize,
      &(scalingFactor),
      outputDescriptor,
      params.cudnn_gSyn
      );

   cudnnHandleError(status, "Convolution run");

   //int scaleFactor = 1;
   //status = cudnnAddTensor(
   //   handle,
   //   CUDNN_ADD_FULL_TENSOR,
   //   &scaleFactor,
   //   outputDescriptor,
   //   params.cudnn_accumGSyn,
   //   &scaleFactor,
   //   outputDescriptor,
   //   params.cudnn_gSyn
   //);

   //assert(status == CUDNN_STATUS_SUCCESS);
   
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

   //TODO make this function handle multiple batches
   //For now, since we mostly use CUDNN, queue many recvs based on batch
   for(int b = 0; b < params.nbatch; b++){
      HyPerLayer_recv_post<<<grid_size, block_size, sharedSize, device->getStream()>>>(params, b);
      handleCallError("Recv from post");
   }
#endif

   return 0;
}

}
