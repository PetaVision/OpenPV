/*
 * RecvPost.cu
 *
 *  Created on: Aug 5, 2014
 *      Author: Sheng Lundquist
 */

#ifndef CUDAUPDATEANNACTIVITY_HPP_
#define CUDAUPDATEANNACTIVITY_HPP_

#include "arch/cuda/CudaKernel.hpp"

#include "arch/cuda/CudaBuffer.hpp"
#include <assert.h>
#include <builtin_types.h>

namespace PVCuda {

// Parameter structure

class CudaUpdateANNActivity : public CudaKernel {
  public:
   struct ANNActivityParams {
      int nbatch;
      int numNeurons;
      int nx;
      int ny;
      int nf;
      int lt;
      int rt;
      int dn;
      int up;

      float const *internalState;
      int numVertices;
      float const *verticesV;
      float const *verticesA;
      float const *slopes;
      float *activity;
   };

   CudaUpdateANNActivity(CudaDevice *inDevice);

   virtual ~CudaUpdateANNActivity();

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

         CudaBuffer *V /*float*/,
         int const numVertices,
         CudaBuffer *verticesV /*float*/,
         CudaBuffer *verticesA /*float*/,
         CudaBuffer *slopes /*float*/,
         CudaBuffer *A /*float*/);

  protected:
   // This is the function that should be overwritten in child classes
   virtual int do_run() override;

  private:
   ANNActivityParams mParams;
};

} /* namespace PVCuda */

#endif // CUDAUPDATEANNACTIVITY_HPP_
