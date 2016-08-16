/*
 * CudaRecvPost.cu
 *
 *  Created on: Aug 5, 2014
 *      Author: Sheng Lundquist
 */

#ifndef CUDARECVPOST_HPP_
#define CUDARECVPOST_HPP_

#include "arch/cuda/CudaKernel.hpp"
#include "arch/cuda/CudaBuffer.hpp"

namespace PVCuda{
#include <builtin_types.h>

//Parameter structure
   struct recv_post_params{
      int nbatch;
      int nxRes; //num post neurons
      int nyRes;
      int nf;
      int nblt; //Border of orig
      int nbrt; //Border of orig
      int nbdn; //Border of orig
      int nbup; //Border of orig

      int preNx;
      int preNy;
      int preNf;
      int preNblt;
      int preNbrt;
      int preNbup;
      int preNbdn;

      int nxp;
      int nyp;
      int nfp;

      int localBufSizeX;
      int localBufSizeY;
      float preToPostScaleX;
      float preToPostScaleY;

      int sy;
      int syp;
      int numPerStride;
      float dt_factor;
      int sharedWeights;

      long* startSourceExtBuf;
      float* preData;
      float* weights;
      float* postGsyn;
#ifdef PV_USE_CUDNN
      float* cudnn_preData;
      float* cudnn_weights;
      float* cudnn_gSyn;
      //float* cudnn_accumGSyn;
      void* cudnn_workspace;
#endif
      int* patch2datalookuptable;

      //Shared num elements
      size_t preBufNum;
      size_t postBufNum;
      size_t weightsBufNum;

      //Warp size of the device
      int warpSize;

      bool preDataLocal;
#ifdef PV_USE_CUDNN
      /* cudnnTensorDescriptor_t */ void* v_inputDescriptor;
      /* cudnnFilterDescriptor_t */   void* v_filterDescriptor;
      /* cudnnTensorDescriptor_t */ void* v_outputDescriptor;
      /* cudnnConvolutionDescriptor_t */ void* v_convDescriptor;
      /* cudnnConvolutionFwdAlgo_t* */ void* v_convAlgo;
      size_t * workspaceSize;
      int manyScaleX;
      int manyScaleY;
      int diffY;
      int diffX;
#endif
   };


class CudaRecvPost : public CudaKernel {
public:
   CudaRecvPost(CudaDevice* inDevice);
   
   virtual ~CudaRecvPost();

   void setArgs(
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
   );

#ifdef PV_USE_CUDNN
   void permuteDatastorePVToCudnn();
   void permuteWeightsPVToCudnn();
   void permuteGSynPVToCudnn(int channel);
   void permuteGSynCudnnToPV(int channel);
#endif

   void set_dt_factor(float new_dt_factor) {params.dt_factor = new_dt_factor;}

protected:
   //This is the function that should be overwritten in child classes
   virtual int do_run();

private:
   recv_post_params params;
}; // end class CudaRecvPost

}  // end namespace PV

#endif /* CLKERNEL_HPP_ */
