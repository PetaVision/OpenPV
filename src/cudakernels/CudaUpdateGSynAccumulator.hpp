/*
 * CudaUpdateGSynAccumulator.hpp
 *
 *  Created on: Nov 14, 2018
 *      Author: Pete Schultz
 */

#ifndef CUDAUPDATEGSYNACCUMULATOR_HPP_
#define CUDAUPDATEGSYNACCUMULATOR_HPP_

#include "arch/cuda/CudaKernel.hpp"

#include "arch/cuda/CudaBuffer.hpp"
#include <builtin_types.h>

namespace PVCuda {

class CudaUpdateGSynAccumulator : public CudaKernel {
  public:
   // Parameter structure
   struct GSynAccumulatorParams {
      int nbatch;
      int numNeurons;
      int numChannels;
      float const *channelCoefficients;
      float const *layerInput;
      float *accumulatedGSyn;
   };

   CudaUpdateGSynAccumulator(CudaDevice *inDevice);

   virtual ~CudaUpdateGSynAccumulator();

   void setArgs(
         int const nbatch,
         int const numNeurons,
         int const numChannels,
         CudaBuffer *channelCoefficients /* float const* */,
         CudaBuffer *layerInput /* float const* */,
         CudaBuffer *accumulatedGSyn /* float* */);

  protected:
   // This is the function that should be overwritten in child classes
   virtual int do_run() override;

  private:
   GSynAccumulatorParams params;
};

} /* namespace PVCuda */

#endif // CUDAUPDATEGSYNACCUMULATOR_HPP_
