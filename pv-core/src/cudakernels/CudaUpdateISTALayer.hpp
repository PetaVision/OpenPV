/*
 * RecvPost.cu
 *
 *  Created on: Aug 5, 2014
 *      Author: Sheng Lundquist
 */

#ifndef CUDAUPDATEISTALAYER_HPP_
#define CUDAUPDATEISTALAYER_HPP_

#include "../arch/cuda/CudaKernel.hpp"
#include "../arch/cuda/CudaBuffer.hpp"
#include <assert.h>

namespace PVCuda{
#include <builtin_types.h>

   //Parameter structur
   struct ISTAParams{
      int nbatch;
      int numNeurons;
      int nx;
      int ny;
      int nf;
      int lt;
      int rt;
      int dn;
      int up;
      int numChannels;

      float * V;
      float Vth;
      float AMax;
      float AMin;
      float AShift;
      float VWidth;
      bool selfInteract;
      double * dtAdapt;
      float tau;
      float * GSynHead;
      float * activity;
   };

class CudaUpdateISTALayer : public CudaKernel {
public:
   CudaUpdateISTALayer(CudaDevice* inDevice);
   
   virtual ~CudaUpdateISTALayer();

   void setArgs(
      const int nbatch,
      const int numNeurons,
      const int nx,
      const int ny,
      const int nf,
      const int lt,
      const int rt,
      const int dn,
      const int up,
      const int numChannels,

      /* float* */ CudaBuffer* V,

      const float Vth,
      const float AMax,
      const float AMin,
      const float AShift,
      const float VWidth,
      const bool selfInteract,
      /* double* */ CudaBuffer* dtAdapt,
      const float tau,

      /* float* */ CudaBuffer* GSynHead,
      /* float* */ CudaBuffer* activity
   );

protected:
   //This is the function that should be overwritten in child classes
   virtual int do_run();

private:
   ISTAParams params;
};

}

#endif /* CLKERNEL_HPP_ */
