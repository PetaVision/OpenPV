#include "CudaUpdateGSynAccumulator.hpp"

namespace PVCuda {

CudaUpdateGSynAccumulator::CudaUpdateGSynAccumulator(CudaDevice *inDevice) : CudaKernel(inDevice) {
   mKernelName = "GSynAccumulator";
}

CudaUpdateGSynAccumulator::~CudaUpdateGSynAccumulator() {}

void CudaUpdateGSynAccumulator::setArgs(
      int const nbatch,
      int const numNeurons,
      int const numChannels,
      CudaBuffer *channelCoefficients /* float const* */,
      CudaBuffer *layerInput /* float const* */,
      CudaBuffer *accumulatedGSyn /* float* */) {
   params.nbatch              = nbatch;
   params.numNeurons          = numNeurons;
   params.numChannels         = numChannels;
   params.channelCoefficients = (float const *)channelCoefficients->getPointer();
   params.layerInput          = (float const *)layerInput->getPointer();
   params.accumulatedGSyn     = (float *)accumulatedGSyn->getPointer();

   setArgsFlag();
}

} // end namespace PVCuda
