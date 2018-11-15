/*
 * RecvPost.cu
 *
 *  Created on: Aug 5, 2014
 *      Author: Sheng Lundquist
 */

#ifndef CUDAUPDATEHYPERLCAINTERNALSTATE_HPP_
#define CUDAUPDATEHYPERLCAINTERNALSTATE_HPP_

#include "arch/cuda/CudaKernel.hpp"

#include "arch/cuda/CudaBuffer.hpp"
#include <builtin_types.h>

namespace PVCuda {

class CudaUpdateHyPerLCAInternalState : public CudaKernel {
  public:
   // Parameter structure
   struct HyPerLCAParams {
      int nbatch;
      int numNeurons;
      int nx;
      int ny;
      int nf;
      int lt;
      int rt;
      int dn;
      int up;

      float *V;
      bool selfInteract;
      double *dtAdapt;
      float tau;
      float const *accumulatedGSyn;
      float const *activity;
   };

   CudaUpdateHyPerLCAInternalState(CudaDevice *inDevice);

   virtual ~CudaUpdateHyPerLCAInternalState();

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

         CudaBuffer *V /* float* */,
         bool const selfInteract,
         CudaBuffer *dtAdapt /* double* */,
         float const tau,

         CudaBuffer *accumulatedGSyn /* float const* */,
         CudaBuffer *activity /* float const* */);

  protected:
   // This is the function that should be overwritten in child classes
   virtual int do_run() override;

  private:
   HyPerLCAParams params;
};

} /* namespace PVCuda */

#endif // CUDAUPDATEHYPERLCAINTERNALSTATE_HPP_
