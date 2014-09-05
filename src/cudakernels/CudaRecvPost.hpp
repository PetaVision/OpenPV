/*
 * RecvPost.cu
 *
 *  Created on: Aug 5, 2014
 *      Author: Sheng Lundquist
 */

#ifndef CUDARECVPOST_HPP
#define CUDARECVPOST_HPP_

#include "../arch/cuda/CudaKernel.hpp"
#include "../arch/cuda/CudaBuffer.hpp"
//#include "../arch/cuda/Cuda3dFloatTextureBuffer.hpp"
//#include "../utils/conversions.h"
//#include "../layers/accumulate_functions.h"

namespace PVCuda{
#include <builtin_types.h>

//Parameter structur
   struct recv_post_params{
      int nxRes; //num post neurons
      int nyRes;
      int nf;
      int nblt; //Border of orig
      int nbrt; //Border of orig
      int nbdn; //Border of orig
      int nbup; //Border of orig
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
      int* patch2datalookuptable;

      //Shared num elements
      size_t preBufNum;
      size_t postBufNum;
      //Warp size of the device
      int warpSize;
   };


class CudaRecvPost : public CudaKernel {
public:
   CudaRecvPost(CudaDevice* inDevice);
   
   virtual ~CudaRecvPost();

   void setArgs(
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

      /* long* */  CudaBuffer* startSourceExtBuf,
      /* float* */ CudaBuffer* preData,
      /* float* */ CudaBuffer* weights,
      /* float* */ CudaBuffer* postGsyn,
      /* int* */   CudaBuffer* patch2datalookuptable
   );

protected:
   //This is the function that should be overwritten in child classes
   virtual int run();

private:
   recv_post_params params;
   //int nxRes; //num post neurons
   //int nyRes;
   //int nf;
   //int nb; //Border of orig
   //int nxp;
   //int nyp;
   //int nfp;

   //int localBufSizeX;
   //int localBufSizeY;
   //float preToPostScaleX;
   //float preToPostScaleY;

   //int sy;
   //int syp;
   //int numPerStride;
   //float dt_factor;
   //int sharedWeights;


   ///* long* */  CudaBuffer* startSourceExtBuf;
   ///* float* */ CudaBuffer* preData;
   ///* float* */ CudaBuffer* weights;
   ///* float* */ CudaBuffer* postGsyn;
   ///* int* */   CudaBuffer* patch2datalookuptable;

};

}

#endif /* CLKERNEL_HPP_ */
