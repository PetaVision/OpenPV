/*
 * RecvPost.cu
 *
 *  Created on: Aug 5, 2014
 *      Author: Sheng Lundquist
 */

#ifndef CUDAUPDATEISTAINTERNALSTATE_HPP_
#define CUDAUPDATEISTAINTERNALSTATE_HPP_

#include "arch/cuda/CudaKernel.hpp"

#include "arch/cuda/CudaBuffer.hpp"
#include <builtin_types.h>

namespace PVCuda {

class CudaUpdateISTAInternalState : public CudaKernel {
  public:
   // Parameter structure
   struct ISTAParams {
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

      float *V;
      float VThresh;
      double *dtAdapt;
      float tau;
      float const *GSynHead;
      float const *activity;
   };

   CudaUpdateISTAInternalState(CudaDevice *inDevice);

   virtual ~CudaUpdateISTAInternalState();

   void setArgs(
         int const nbatch,
         int const numNeurons,
         int const nx,
         int const ny,
         int const nf,
         int const lt,
         int const rt,
         int const dn,
         int const up,
         int const numChannels,

         CudaBuffer *V /* float* */,
         float const VThresh,
         CudaBuffer *dtAdapt /* double* */,
         float const tau,

         CudaBuffer *GSynHead /* float const* */,
         CudaBuffer *activity /* float const* */);

  protected:
   // This is the function that should be overwritten in child classes
   virtual int do_run() override;

  private:
   ISTAParams params;
};

} /* namespace PVCuda */

#endif // CUDAUPDATEISTAINTERNALSTATE_HPP_
