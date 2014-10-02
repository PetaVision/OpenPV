/*
 * RecvPost.cu
 *
 *  Created on: Aug 5, 2014
 *      Author: Sheng Lundquist
 */

#ifndef CUDAUPDATEHYPERLCALAYER_HPP_
#define CUDAUPDATEHYPERLCALAYER_HPP_

#include "../arch/cuda/CudaKernel.hpp"
#include "../arch/cuda/CudaBuffer.hpp"
#include <assert.h>

namespace PVCuda{
#include <builtin_types.h>

   //Parameter structur
   struct HyPerLCAParams{
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
      float dt_tau;
      float * GSynHead;
      float * activity;
   };

class CudaUpdateHyPerLCALayer : public CudaKernel {
public:
   CudaUpdateHyPerLCALayer(CudaDevice* inDevice);
   
   virtual ~CudaUpdateHyPerLCALayer();

   void setArgs(
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
      const float dt_tau,

      /* float* */ CudaBuffer* GSynHead,
      /* float* */ CudaBuffer* activity
   );

   void setDtTau(float dt_tau){params.dt_tau = dt_tau;}

protected:
   //This is the function that should be overwritten in child classes
   virtual int do_run();

private:
   HyPerLCAParams params;
};

}

#endif /* CLKERNEL_HPP_ */
