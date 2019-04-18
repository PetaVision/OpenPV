#include "CudaRecvPost.hpp"
#include "arch/cuda/cuda_util.hpp"
#include "conversions.hcu"
#include "utils/PVAssert.hpp"
#include "utils/PVLog.hpp"
#include <cmath>
#include <sstream>

namespace PVCuda {

#ifdef PV_USE_CUDNN
#include <cudnn.h>

#endif // PV_USE_CUDNN

CudaRecvPost::CudaRecvPost(CudaDevice *inDevice) : CudaKernel(inDevice) {
   kernelName = "CudaRecvPost";
}

CudaRecvPost::~CudaRecvPost() {
#ifdef PV_USE_CUDNN
   if (params.v_inputDescriptor) {
      cudnnTensorDescriptor_t inputDescriptor = (cudnnTensorDescriptor_t)params.v_inputDescriptor;
      cudnnDestroyTensorDescriptor(inputDescriptor);
   }
   if (params.v_filterDescriptor) {
      cudnnFilterDescriptor_t filterDescriptor = (cudnnFilterDescriptor_t)params.v_filterDescriptor;
      cudnnDestroyFilterDescriptor(filterDescriptor);
   }
   if (params.v_outputDescriptor) {
      cudnnTensorDescriptor_t outputDescriptor = (cudnnTensorDescriptor_t)params.v_outputDescriptor;
      cudnnDestroyTensorDescriptor(outputDescriptor);
   }
   if (params.v_convDescriptor) {
      cudnnConvolutionDescriptor_t convDescriptor =
            (cudnnConvolutionDescriptor_t)params.v_convDescriptor;
      cudnnDestroyConvolutionDescriptor(convDescriptor);
   }
   if (params.v_convAlgo) {
      cudnnConvolutionFwdAlgo_t *convAlgo = (cudnnConvolutionFwdAlgo_t *)params.v_convAlgo;
      delete convAlgo;
   }
   if (params.cudnn_workspace) {
      handleError(cudaFree(params.cudnn_workspace), "Freeing workspace pointer");
   }
   if (params.workspaceSize) {
      delete params.workspaceSize;
   }
#endif // PV_USE_CUDNN
}

void CudaRecvPost::setArgs(
      const int nbatch,
      const int nxRes, // num post neurons
      const int nyRes,
      const int nf,

      const int nblt, // Border of orig
      const int nbrt, // Border of orig
      const int nbdn, // Border of orig
      const int nbup, // Border of orig

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

      const float preToPostScaleX,
      const float preToPostScaleY,

      const int sy,
      const int syp,
      const int numPerStride,
      const float dt_factor,
      const int sharedWeights,

      /* long* */ CudaBuffer *startSourceExtBuf,
      /* float* */ CudaBuffer *preData,
      /* float* */ CudaBuffer *weights,
      /* float* */ CudaBuffer *postGsyn,
#ifdef PV_USE_CUDNN
      /* float* */ CudaBuffer *cudnn_preData,
      /* float* */ CudaBuffer *cudnn_weights,
      /* float* */ CudaBuffer *cudnn_gSyn,
#endif // PV_USE_CUDNN
      /* int* */ CudaBuffer *patch2datalookuptable) {
   params.nbatch = nbatch;
   params.nxRes  = nxRes;
   params.nyRes  = nyRes;
   params.nf     = nf;

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

   params.preToPostScaleX = preToPostScaleX;
   params.preToPostScaleY = preToPostScaleY;

   params.sy            = sy;
   params.syp           = syp;
   params.numPerStride  = numPerStride;
   params.dt_factor     = dt_factor;
   params.sharedWeights = sharedWeights;

   params.startSourceExtBuf = (long *)startSourceExtBuf->getPointer();
   params.preData           = (float *)preData->getPointer();
   params.weights           = (float *)weights->getPointer();
   params.postGsyn          = (float *)postGsyn->getPointer();
#ifdef PV_USE_CUDNN
   params.cudnn_weights = (float *)cudnn_weights->getPointer();
   params.cudnn_preData = (float *)cudnn_preData->getPointer();
   params.cudnn_gSyn    = (float *)cudnn_gSyn->getPointer();
#endif // PV_USE_CUDNN
   params.patch2datalookuptable = (int *)patch2datalookuptable->getPointer();

   params.warpSize = device->get_warp_size();

#ifdef PV_USE_CUDNN
   // CUDNN code
   // Calculate how much space is left on the gpu for the workspace memory
   // Do not add to device's count since there might be more than one kernel that needs workspace
   // memory
   size_t workspaceMem = device->getMemory() / device->getNumConvKernels();

   int strideX, strideY;
   int actualXBorder, actualYBorder;
   pvAssert(params.preNblt == params.preNbrt);
   pvAssert(params.preNbup == params.preNbdn);
   // One to many case
   if (preToPostScaleX < 1) {
      float fmanyScale = (float)1 / params.preToPostScaleX;
      // Make sure manyScale is an actual integer
      pvAssert(std::ceil(fmanyScale) == fmanyScale);
      params.manyScaleX = fmanyScale;
      fmanyScale        = (float)1 / params.preToPostScaleY;
      pvAssert(std::ceil(fmanyScale) == fmanyScale);
      params.manyScaleY = fmanyScale;
      strideX           = 1;
      strideY           = 1;

      // Patch sizes must be odd multiple of many
      if (nxp % 2 == 0 || nyp % 2 == 0) {
         ErrorLog().printf(
               "cuDNN: Running on a one to many connection with CUDNN must have patch size (%d, "
               "%d) be an odd muliple of many (%d, %d)\n",
               nxp * params.manyScaleX,
               nyp * params.manyScaleY,
               params.manyScaleX,
               params.manyScaleY);
      }

      // There's the case where the border of pre is made bigger through other connections. Need to
      // calculate difference
      // between current recv border and actual recv border
      // This is calculating what the border would be if this was a one to one connection
      actualXBorder = params.nxp / 2;
      actualYBorder = params.nyp / 2;
   }
   // Many to one or one to one case
   else {
      params.manyScaleX = 1;
      params.manyScaleY = 1;
      pvAssert(std::ceil(preToPostScaleX) == preToPostScaleX);
      pvAssert(std::ceil(preToPostScaleY) == preToPostScaleY);
      strideX = preToPostScaleX;
      strideY = preToPostScaleY;

      // There's the case where the border of pre is made bigger through other connections. Need to
      // calculate difference
      // between current recv border and actual recv border
      actualXBorder = (params.nxp - params.preToPostScaleX) / 2;
      actualYBorder = (params.nyp - params.preToPostScaleY) / 2;
   }

   // diffX is positive value of cropping
   params.diffX = params.preNblt - actualXBorder;
   params.diffY = params.preNbup - actualYBorder;

   // Set up pre descriptor
   cudnnTensorDescriptor_t inputDescriptor;
   cudnnStatus_t status = cudnnCreateTensorDescriptor(&inputDescriptor);
   cudnnHandleError(status, "Create input tensor descriptor");

   status = cudnnSetTensor4dDescriptor(
         inputDescriptor,
         CUDNN_TENSOR_NCHW,
         CUDNN_DATA_FLOAT,
         nbatch, // Number of images
         params.preNf, // Number of feature maps per image
         params.preNy + params.preNbup + params.preNbdn
               - 2 * params.diffY, // Height of each feature map
         params.preNx + params.preNblt + params.preNbrt
               - 2 * params.diffX); // Width of each feature map
   if (status != CUDNN_STATUS_SUCCESS) {
      switch (status) {
         case CUDNN_STATUS_BAD_PARAM: Fatal().printf("cuDNN bad parameter\n"); break;
         default: Fatal().printf("cuDNN unknown error code %d\n", status);
      }
      pvAssert(0);
   }
   cudnnHandleError(status, "Set input tensor descriptor");
   params.v_inputDescriptor = (void *)inputDescriptor;

   // Set up filter descriptor
   cudnnFilterDescriptor_t filterDescriptor;
   status = cudnnCreateFilterDescriptor(&filterDescriptor);
   cudnnHandleError(status, "Create filter tensor descriptor");
#if CUDNN_MAJOR >= 5
   status = cudnnSetFilter4dDescriptor(
         filterDescriptor,
         CUDNN_DATA_FLOAT,
         CUDNN_TENSOR_NCHW,
         params.nf * params.manyScaleX * params.manyScaleY, // Number of output feature maps. For
         // one to many, output feature maps are
         // repeated for each kernel
         params.nfp, // Number of input feature maps
         params.nyp, // Height of each filter
         params.nxp); // Width of each filter
#elif CUDNN_MAJOR == 4
   status = cudnnSetFilter4dDescriptor(
         filterDescriptor,
         CUDNN_DATA_FLOAT,
         params.nf * params.manyScaleX * params.manyScaleY, // Number of output feature maps. For
         // one to many, output feature maps are
         // repeated for each kernel
         params.nfp, // Number of input feature maps
         params.nyp, // Height of each filter
         params.nxp); // Width of each filter
#else
#error The cuDNN version is required to be either v4 or greater.\n
#endif
   cudnnHandleError(status, "Set filter tensor descriptor");
   params.v_filterDescriptor = (void *)filterDescriptor;

   // Set convolution descriptor
   cudnnConvolutionDescriptor_t convDescriptor;
   status = cudnnCreateConvolutionDescriptor(&convDescriptor);
   cudnnHandleError(status, "Create convolution tensor descriptor");
   status = cudnnSetConvolution2dDescriptor(
         convDescriptor,
         0,
         0, // zero-padding height and width
         strideY, // Vertical filter stride
         strideX, // Horizontal filter stride
         1,
         1, // upscale the input in x/y direction
         CUDNN_CONVOLUTION
#if CUDNN_MAJOR >= 6
         ,
         CUDNN_DATA_FLOAT
#endif
         );
   cudnnHandleError(status, "Set convolution tensor descriptor");
   params.v_convDescriptor = (void *)convDescriptor;

   // Query output layout and check with PV layout
   int out_n, out_c, out_h, out_w;
   status = cudnnGetConvolution2dForwardOutputDim(
         convDescriptor,
         inputDescriptor,
         filterDescriptor,
         &out_n, // num images
         &out_c, // num output features
         &out_h, // output height
         &out_w); // output width
   cudnnHandleError(status, "Get output tensor descriptor");

   // Make sure dimensions match up with PV layer
   if (out_n != nbatch || out_h != nyRes / params.manyScaleY || out_w != nxRes / params.manyScaleX
       || out_c != nf * params.manyScaleX * params.manyScaleY) {
      std::stringstream errmsg("");
      errmsg << "CUDNN:: Dimensions don't match: \n";
      errmsg << "Dimensions of output tensor (n, y, x, f): " << out_n << ", " << out_h << ", "
             << out_w << ", " << out_c << "\n";
      errmsg << "Scaled dimensions of output PV layer (n, y, x, f): " << nbatch << ", "
             << nyRes / params.manyScaleY << ", " << nxRes / params.manyScaleX << ", "
             << nf * params.manyScaleX * params.manyScaleY << "\n";
      errmsg << "Actual dimensions of output PV layer (n, y, x, f): " << nbatch << ", " << nyRes
             << ", " << nxRes << ", " << nf << "\n";
      Fatal() << errmsg.str() << std::endl;
   }

   // Set up output descriptor
   cudnnTensorDescriptor_t outputDescriptor;
   status = cudnnCreateTensorDescriptor(&outputDescriptor);
   cudnnHandleError(status, "Create output tensor descriptor");
   status = cudnnSetTensor4dDescriptor(
         outputDescriptor,
         CUDNN_TENSOR_NCHW,
         CUDNN_DATA_FLOAT,
         nbatch, // Number of images
         nf * params.manyScaleX * params.manyScaleY, // Number of feature maps per image
         nyRes / params.manyScaleY, // ny restricted
         nxRes / params.manyScaleX); // nx restricted
   cudnnHandleError(status, "Set output tensor descriptor");
   params.v_outputDescriptor = (void *)outputDescriptor;

   // Calculate and set up best forward conv algorithm to use
   cudnnHandle_t handle                = (cudnnHandle_t)device->getCudnnHandle();
   cudnnConvolutionFwdAlgo_t *convAlgo = new cudnnConvolutionFwdAlgo_t();

   status = cudnnGetConvolutionForwardAlgorithm(
         handle,
         inputDescriptor,
         filterDescriptor,
         convDescriptor,
         outputDescriptor,
         CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
         workspaceMem,
         convAlgo);
   cudnnHandleError(status, "Get convolution forward algorithm");
   params.v_convAlgo = (void *)convAlgo;

   // Based on algorithm, allocate workspace memory for GPU
   size_t *temp = new size_t();
   status       = cudnnGetConvolutionForwardWorkspaceSize(
         handle,
         inputDescriptor,
         filterDescriptor,
         convDescriptor,
         outputDescriptor,
         *convAlgo,
         temp);
   params.workspaceSize = temp;
   cudnnHandleError(status, "Get convolution forward workspace size");

   // Allocate workspace based on size
   handleError(
         cudaMalloc(&params.cudnn_workspace, *params.workspaceSize), "Cudnn workspace cudaMalloc");

#endif // PV_USE_CUDNN

   setArgsFlag();
}

int CudaRecvPost::do_run() {

#ifdef PV_USE_CUDNN
   cudnnHandle_t handle                     = (cudnnHandle_t)device->getCudnnHandle();
   cudnnTensorDescriptor_t inputDescriptor  = (cudnnTensorDescriptor_t)params.v_inputDescriptor;
   cudnnFilterDescriptor_t filterDescriptor = (cudnnFilterDescriptor_t)params.v_filterDescriptor;
   cudnnTensorDescriptor_t outputDescriptor = (cudnnTensorDescriptor_t)params.v_outputDescriptor;
   cudnnConvolutionDescriptor_t convDescriptor =
         (cudnnConvolutionDescriptor_t)params.v_convDescriptor;
   cudnnConvolutionFwdAlgo_t *convAlgo = (cudnnConvolutionFwdAlgo_t *)params.v_convAlgo;

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
         params.cudnn_gSyn);

   cudnnHandleError(status, "Convolution run");
#endif // PV_USE_CUDNN

   return 0;
}

#ifdef PV_USE_CUDNN
void CudaRecvPost::permuteDatastorePVToCudnn() {
   // Ext pre activity
   int ny     = params.preNy + params.preNbup + params.preNbdn;
   int nx     = params.preNx + params.preNblt + params.preNbrt;
   int nf     = params.preNf;
   int nbatch = params.nbatch;

   // Calculate grid and work size
   int numNeurons = nbatch * ny * nx * nf;
   int blockSize  = device->get_max_threads();
   // Ceil to get all weights
   int gridSize = ceil((float)numNeurons / blockSize);

   device->syncDevice();

   callPermuteDatastorePVToCudnnKernel(
         gridSize,
         blockSize,
         params.preData,
         params.cudnn_preData,
         nbatch,
         ny,
         nx,
         nf,
         params.diffX,
         params.diffY);
   handleCallError("Permute PV to CUDNN");
}

void CudaRecvPost::permuteGSynPVToCudnn(int channel) {
   // Res post activity
   int ny     = params.nyRes;
   int nx     = params.nxRes;
   int nf     = params.nf;
   int nbatch = params.nbatch;

   // Calculate grid and work size
   int numNeurons       = nbatch * ny * nx * nf;
   float *gSynPatchHead = &(params.postGsyn[numNeurons * channel]);

   int blockSize = device->get_max_threads();
   // Ceil to get all weights
   int gridSize = std::ceil((float)numNeurons / (float)blockSize);
   callPermuteGSynPVToCudnnKernel(
         gridSize,
         blockSize,
         gSynPatchHead,
         params.cudnn_gSyn,
         nbatch,
         ny,
         nx,
         nf,
         params.manyScaleX,
         params.manyScaleY);
   handleCallError("Permute GSyn PV to CUDNN");
}

void CudaRecvPost::permuteGSynCudnnToPV(int channel) {
   // Res post activity
   int ny     = params.nyRes;
   int nx     = params.nxRes;
   int nf     = params.nf;
   int nbatch = params.nbatch;

   // Calculate grid and work size
   int numNeurons       = nbatch * ny * nx * nf;
   float *gSynPatchHead = &(params.postGsyn[numNeurons * channel]);

   int blockSize = device->get_max_threads();
   // Ceil to get all weights
   int gridSize = ceil((float)numNeurons / blockSize);
   callPermuteGSynCudnnToPVKernel(
         gridSize,
         blockSize,
         gSynPatchHead,
         params.cudnn_gSyn,
         nbatch,
         ny,
         nx,
         nf,
         params.manyScaleX,
         params.manyScaleY);
   handleCallError("Permute GSyn CUDNN to PV");
}

#endif // PV_USE_CUDNN

} // namespace PVCuda
