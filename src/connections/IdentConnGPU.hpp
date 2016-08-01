#ifndef _GPUIDENTCONN_HPP_
#define _GPUIDENTCONN_HPP_

#include "../layers/ANNLayerGPU.hpp"
#include "HyPerConnGPU.hpp"

namespace GPULCA {
class IdentConnGPU : public HyPerConnGPU {
 public:
  IdentConnGPU(const char *name, PV::HyPerCol *hc);
	virtual ~IdentConnGPU();
  virtual int communicateInitInfo();
  virtual int deliver();
  virtual int allocateDataStructures();

 protected:
  IdentConnGPU();
	
 private:
  void initialize_base();

 private:
  DenseMatrix<pvdata_t> buf;
  PVCudaWrapper<pvwdata_t> *PreNHWC;

	/*  CUDA handle */
  cudnnHandle_t cudnnHandle;
  cudnnTensorDescriptor_t cudnnTensorDescriptorPreNHWC, cudnnTensorDescriptorPre;
};
}

#endif  // _IDENTCONNGPU_HPP_
