/*
 * RecvPost.cu
 *
 *  Created on: Aug 5, 2014
 *      Author: Sheng Lundquist
 */

#ifndef CUDAUPDATEMOMENTUMLCAINTERNALSTATE_HPP_
#define CUDAUPDATEMOMENTUMLCAINTERNALSTATE_HPP_

#include "arch/cuda/CudaKernel.hpp"

#include "arch/cuda/CudaBuffer.hpp"
#include <builtin_types.h>

namespace PVCuda {

class CudaUpdateMomentumLCAInternalState : public CudaKernel {
  public:
   // Parameter structure
   struct MomentumLCAParams {
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
      float *prevDrive;
      bool selfInteract;
      double *dtAdapt;
      float tau;
      float LCAMomentumRate;
      float const *GSynHead;
      float const *activity;
   };

   CudaUpdateMomentumLCAInternalState(CudaDevice *inDevice);

   virtual ~CudaUpdateMomentumLCAInternalState();

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
         CudaBuffer *prevDrive /* float* */,
         bool const selfInteract,
         CudaBuffer *dtAdapt /* double* */,
         float const tau,
         float const LCAMomentumRate,

         CudaBuffer *GSynHead /* float const* */,
         CudaBuffer *activity /* float const* */);

  protected:
   // This is the function that should be overwritten in child classes
   virtual int do_run() override;

  private:
   MomentumLCAParams params;
};

} /* namespace PVCuda */

#endif // CUDAUPDATEMOMENTUMLCAINTERNALSTATE_HPP_
