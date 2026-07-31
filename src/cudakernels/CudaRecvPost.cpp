#include "CudaRecvPost.hpp"
#include "arch/cuda/cuda_util.hpp"
#include "utils/PVAssert.hpp"
#include "utils/PVLog.hpp"
#include <cmath>
#include <sstream>

namespace PVCuda {

#ifdef PV_USE_CUDNN
#include <cudnn.h>

#endif // PV_USE_CUDNN

CudaRecvPost::CudaRecvPost(CudaDevice *inDevice) : CudaKernel(inDevice) {
   mKernelName = "CudaRecvPost";
}

CudaRecvPost::~CudaRecvPost() {
#ifdef PV_USE_CUDNN
   if (mParams.v_inputDescriptor) {
      cudnnTensorDescriptor_t inputDescriptor = (cudnnTensorDescriptor_t)mParams.v_inputDescriptor;
      cudnnDestroyTensorDescriptor(inputDescriptor);
   }
   if (mParams.v_filterDescriptor) {
      cudnnFilterDescriptor_t filterDescriptor =
            (cudnnFilterDescriptor_t)mParams.v_filterDescriptor;
      cudnnDestroyFilterDescriptor(filterDescriptor);
   }
   if (mParams.v_outputDescriptor) {
      cudnnTensorDescriptor_t outputDescriptor =
            (cudnnTensorDescriptor_t)mParams.v_outputDescriptor;
      cudnnDestroyTensorDescriptor(outputDescriptor);
   }
   if (mParams.v_convDescriptor) {
      cudnnConvolutionDescriptor_t convDescriptor =
            (cudnnConvolutionDescriptor_t)mParams.v_convDescriptor;
      cudnnDestroyConvolutionDescriptor(convDescriptor);
   }
   if (mParams.v_convAlgo) {
      cudnnConvolutionFwdAlgo_t *convAlgo = (cudnnConvolutionFwdAlgo_t *)mParams.v_convAlgo;
      delete convAlgo;
   }
   if (mParams.cudnn_workspace) {
      handleError(cudaFree(mParams.cudnn_workspace), "Freeing workspace pointer");
   }
   if (mParams.workspaceSize) {
      delete mParams.workspaceSize;
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
   mParams.nbatch = nbatch;
   mParams.nxRes  = nxRes;
   mParams.nyRes  = nyRes;
   mParams.nf     = nf;

   mParams.nblt = nblt;
   mParams.nbrt = nbrt;
   mParams.nbdn = nbdn;
   mParams.nbup = nbup;

   mParams.preNx   = preNx;
   mParams.preNy   = preNy;
   mParams.preNf   = preNf;
   mParams.preNblt = preNblt;
   mParams.preNbrt = preNbrt;
   mParams.preNbup = preNbup;
   mParams.preNbdn = preNbdn;

   mParams.nxp = nxp;
   mParams.nyp = nyp;
   mParams.nfp = nfp;

   mParams.preToPostScaleX = preToPostScaleX;
   mParams.preToPostScaleY = preToPostScaleY;

   mParams.sy            = sy;
   mParams.syp           = syp;
   mParams.numPerStride  = numPerStride;
   mParams.dt_factor     = dt_factor;
   mParams.sharedWeights = sharedWeights;

   mParams.startSourceExtBuf = (long *)startSourceExtBuf->getPointer();
   mParams.preData           = (float *)preData->getPointer();
   mParams.weights           = (float *)weights->getPointer();
   mParams.postGsyn          = (float *)postGsyn->getPointer();
#ifdef PV_USE_CUDNN
   mParams.cudnn_weights = (float *)cudnn_weights->getPointer();
   mParams.cudnn_preData = (float *)cudnn_preData->getPointer();
   mParams.cudnn_gSyn    = (float *)cudnn_gSyn->getPointer();
#endif // PV_USE_CUDNN
   mParams.patch2datalookuptable = (int *)patch2datalookuptable->getPointer();

   mParams.warpSize = mDevice->get_warp_size();

#ifdef PV_USE_CUDNN
   int strideX, strideY;
   int actualXBorder, actualYBorder;
   pvAssert(mParams.preNblt == mParams.preNbrt);
   pvAssert(mParams.preNbup == mParams.preNbdn);
   // One to many case
   if (preToPostScaleX < 1.0f) {
      float fmanyScale = 1.0f / mParams.preToPostScaleX;
      // Make sure manyScale is an actual integer
      pvAssert(std::ceil(fmanyScale) == fmanyScale);
      mParams.manyScaleX = (int)fmanyScale;
      fmanyScale        = 1.0f / mParams.preToPostScaleY;
      pvAssert(std::ceil(fmanyScale) == fmanyScale);
      mParams.manyScaleY = (int)fmanyScale;
      strideX           = 1;
      strideY           = 1;

      // Patch sizes must be odd multiple of many
      if (nxp % 2 == 0 || nyp % 2 == 0) {
         ErrorLog().printf(
               "cuDNN: Running on a one to many connection with CUDNN must have patch size (%d, "
               "%d) be an odd muliple of many (%d, %d)\n",
               nxp * mParams.manyScaleX,
               nyp * mParams.manyScaleY,
               mParams.manyScaleX,
               mParams.manyScaleY);
      }

      // There's the case where the border of pre is made bigger through other connections. Need to
      // calculate difference
      // between current recv border and actual recv border
      // This is calculating what the border would be if this was a one to one connection
      actualXBorder = mParams.nxp / 2;
      actualYBorder = mParams.nyp / 2;
   }
   // Many to one or one to one case
   else {
      mParams.manyScaleX = 1;
      mParams.manyScaleY = 1;
      pvAssert(std::ceil(preToPostScaleX) == preToPostScaleX);
      pvAssert(std::ceil(preToPostScaleY) == preToPostScaleY);
      strideX = (int)preToPostScaleX;
      strideY = (int)preToPostScaleY;

      // There's the case where the border of pre is made bigger through other connections. Need to
      // calculate difference
      // between current recv border and actual recv border
      actualXBorder = (mParams.nxp - (int)mParams.preToPostScaleX) / 2;
      actualYBorder = (mParams.nyp - (int)mParams.preToPostScaleY) / 2;
   }

   // diffX is positive value of cropping
   mParams.diffX = mParams.preNblt - actualXBorder;
   mParams.diffY = mParams.preNbup - actualYBorder;

   // Set up pre descriptor
   cudnnTensorDescriptor_t inputDescriptor;
   cudnnStatus_t status = cudnnCreateTensorDescriptor(&inputDescriptor);
   cudnnHandleError(status, "Create input tensor descriptor");

   status = cudnnSetTensor4dDescriptor(
         inputDescriptor,
         CUDNN_TENSOR_NCHW,
         CUDNN_DATA_FLOAT,
         nbatch, // Number of images
         mParams.preNf, // Number of feature maps per image
         mParams.preNy + mParams.preNbup + mParams.preNbdn
               - 2 * mParams.diffY, // Height of each feature map
         mParams.preNx + mParams.preNblt + mParams.preNbrt
               - 2 * mParams.diffX); // Width of each feature map
   if (status != CUDNN_STATUS_SUCCESS) {
      switch (status) {
         case CUDNN_STATUS_BAD_PARAM: Fatal().printf("cuDNN bad parameter\n"); break;
         default: Fatal().printf("cuDNN unknown error code %d\n", status);
      }
      pvAssert(0);
   }
   cudnnHandleError(status, "Set input tensor descriptor");
   mParams.v_inputDescriptor = (void *)inputDescriptor;

   // Set up filter descriptor
   cudnnFilterDescriptor_t filterDescriptor;
   status = cudnnCreateFilterDescriptor(&filterDescriptor);
   cudnnHandleError(status, "Create filter tensor descriptor");
#if CUDNN_MAJOR >= 5
   status = cudnnSetFilter4dDescriptor(
         filterDescriptor,
         CUDNN_DATA_FLOAT,
         CUDNN_TENSOR_NCHW,
         mParams.nf * mParams.manyScaleX * mParams.manyScaleY, // Number of output feature maps.
         // For one to many, output feature maps are repeated for each kernel
         mParams.nfp, // Number of input feature maps
         mParams.nyp, // Height of each filter
         mParams.nxp); // Width of each filter
#elif CUDNN_MAJOR == 4
   status = cudnnSetFilter4dDescriptor(
         filterDescriptor,
         CUDNN_DATA_FLOAT,
         mParams.nf * mParams.manyScaleX * mParams.manyScaleY, // Number of output feature maps. For
         // one to many, output feature maps are
         // repeated for each kernel
         mParams.nfp, // Number of input feature maps
         mParams.nyp, // Height of each filter
         mParams.nxp); // Width of each filter
#else
#error The cuDNN version is required to be either v4 or greater.\n
#endif
   cudnnHandleError(status, "Set filter tensor descriptor");
   mParams.v_filterDescriptor = (void *)filterDescriptor;

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
   mParams.v_convDescriptor = (void *)convDescriptor;

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
   if (out_n != nbatch || out_h != nyRes / mParams.manyScaleY || out_w != nxRes / mParams.manyScaleX
       || out_c != nf * mParams.manyScaleX * mParams.manyScaleY) {
      std::stringstream errmsg("");
      errmsg << "CUDNN:: Dimensions don't match: \n";
      errmsg << "Dimensions of output tensor (n, y, x, f): " << out_n << ", " << out_h << ", "
             << out_w << ", " << out_c << "\n";
      errmsg << "Scaled dimensions of output PV layer (n, y, x, f): " << nbatch << ", "
             << nyRes / mParams.manyScaleY << ", " << nxRes / mParams.manyScaleX << ", "
             << nf * mParams.manyScaleX * mParams.manyScaleY << "\n";
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
         nf * mParams.manyScaleX * mParams.manyScaleY, // Number of feature maps per image
         nyRes / mParams.manyScaleY, // ny restricted
         nxRes / mParams.manyScaleX); // nx restricted
   cudnnHandleError(status, "Set output tensor descriptor");
   mParams.v_outputDescriptor = (void *)outputDescriptor;

   // Calculate and set up best forward conv algorithm to use
   cudnnHandle_t handle                        = (cudnnHandle_t)mDevice->getCudnnHandle();
   cudnnConvolutionFwdAlgoPerf_t *convAlgoPerf = new cudnnConvolutionFwdAlgoPerf_t();


   int returnedAlgoCount;
   status = cudnnGetConvolutionForwardAlgorithm_v7(
         handle,
         inputDescriptor,
         filterDescriptor,
         convDescriptor,
         outputDescriptor,
         1,
         &returnedAlgoCount,
         convAlgoPerf);
   cudnnHandleError(status, "Get convolution forward algorithm");
   mParams.v_convAlgo = (void *)(&convAlgoPerf->algo);

   // Based on algorithm, allocate workspace memory for GPU
   size_t *temp = new size_t();
   status       = cudnnGetConvolutionForwardWorkspaceSize(
         handle,
         inputDescriptor,
         filterDescriptor,
         convDescriptor,
         outputDescriptor,
         convAlgoPerf->algo,
         temp);
   mParams.workspaceSize = temp;
   cudnnHandleError(status, "Get convolution forward workspace size");

   // Allocate workspace based on size
   handleError(
         cudaMalloc(&mParams.cudnn_workspace,
         *mParams.workspaceSize),
         "Cudnn workspace cudaMalloc");

#endif // PV_USE_CUDNN

   setArgsFlag();
}

int CudaRecvPost::do_run() {

#ifdef PV_USE_CUDNN
   cudnnHandle_t handle                     = (cudnnHandle_t)mDevice->getCudnnHandle();
   cudnnTensorDescriptor_t inputDescriptor  = (cudnnTensorDescriptor_t)mParams.v_inputDescriptor;
   cudnnFilterDescriptor_t filterDescriptor = (cudnnFilterDescriptor_t)mParams.v_filterDescriptor;
   cudnnTensorDescriptor_t outputDescriptor = (cudnnTensorDescriptor_t)mParams.v_outputDescriptor;
   cudnnConvolutionDescriptor_t convDescriptor =
         (cudnnConvolutionDescriptor_t)mParams.v_convDescriptor;
   cudnnConvolutionFwdAlgo_t *convAlgo = (cudnnConvolutionFwdAlgo_t *)mParams.v_convAlgo;

   float scalingFactor = 1;

   cudnnStatus_t status = cudnnConvolutionForward(
         handle,
         &(scalingFactor),
         inputDescriptor,
         mParams.cudnn_preData,
         filterDescriptor,
         mParams.cudnn_weights,
         convDescriptor,
         *convAlgo,
         mParams.cudnn_workspace,
         *mParams.workspaceSize,
         &(scalingFactor),
         outputDescriptor,
         mParams.cudnn_gSyn);

   cudnnHandleError(status, "Convolution run");
#endif // PV_USE_CUDNN

   return 0;
}

#ifdef PV_USE_CUDNN
void CudaRecvPost::permuteDatastorePVToCudnn() {
   // Ext pre activity
   int ny     = mParams.preNy + mParams.preNbup + mParams.preNbdn;
   int nx     = mParams.preNx + mParams.preNblt + mParams.preNbrt;
   int nf     = mParams.preNf;
   int nbatch = mParams.nbatch;

   // Calculate grid and work size
   int numNeurons = nbatch * ny * nx * nf;
   int blockSize  = mDevice->get_max_threads();
   // Ceil to get all weights
   int gridSize = (int)std::ceil((float)numNeurons / (float)blockSize);

   callPermuteDatastorePVToCudnnKernel(
         gridSize,
         blockSize,
         mParams.preData,
         mParams.cudnn_preData,
         nbatch,
         ny,
         nx,
         nf,
         mParams.diffX,
         mParams.diffY);
   handleCallError("Permute PV to CUDNN");
}

void CudaRecvPost::permuteGSynPVToCudnn(int channel) {
   // Res post activity
   int ny     = mParams.nyRes;
   int nx     = mParams.nxRes;
   int nf     = mParams.nf;
   int nbatch = mParams.nbatch;

   // Calculate grid and work size
   int numNeurons       = nbatch * ny * nx * nf;
   float *gSynPatchHead = &(mParams.postGsyn[numNeurons * channel]);

   int blockSize = mDevice->get_max_threads();
   // Ceil to get all weights
   int gridSize = (int)std::ceil((float)numNeurons / (float)blockSize);
   callPermuteGSynPVToCudnnKernel(
         gridSize,
         blockSize,
         gSynPatchHead,
         mParams.cudnn_gSyn,
         nbatch,
         ny,
         nx,
         nf,
         mParams.manyScaleX,
         mParams.manyScaleY);
   handleCallError("Permute GSyn PV to CUDNN");
}

void CudaRecvPost::permuteGSynCudnnToPV(int channel) {
   // Res post activity
   int ny     = mParams.nyRes;
   int nx     = mParams.nxRes;
   int nf     = mParams.nf;
   int nbatch = mParams.nbatch;

   // Calculate grid and work size
   int numNeurons       = nbatch * ny * nx * nf;
   float *gSynPatchHead = &(mParams.postGsyn[numNeurons * channel]);

   int blockSize = mDevice->get_max_threads();
   // Ceil to get all weights
   int gridSize = (int)std::ceil((float)numNeurons / (float)blockSize);
   callPermuteGSynCudnnToPVKernel(
         gridSize,
         blockSize,
         gSynPatchHead,
         mParams.cudnn_gSyn,
         nbatch,
         ny,
         nx,
         nf,
         mParams.manyScaleX,
         mParams.manyScaleY);
   handleCallError("Permute GSyn CUDNN to PV");
}

#endif // PV_USE_CUDNN

} // namespace PVCuda
