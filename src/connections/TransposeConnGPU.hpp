#ifndef _TRANSPOSECONNGPU_HPP_
#define _TRANSPOSECONNGPU_HPP_

#include <cudnn.h>
#include "../arch/cuda/CudaMatrix.hpp"
#include "../arch/cuda/CudaUtility.hpp"
#include "HyPerConnGPU.hpp"


namespace GPULCA {
class TransposeConnGPU : public HyPerConnGPU {
 public:
  TransposeConnGPU();
  TransposeConnGPU(const char* name, PV::HyPerCol* hc);
  virtual ~TransposeConnGPU();
  virtual int communicateInitInfo();
  virtual int allocateDataStructures();
	virtual int deliver();
  HyPerConnGPU* getOriginalConn() { return originalConn; }
	

 protected:
  virtual int findCudnnAlgo();

 private:
  char* originalConnName;
  HyPerConnGPU* originalConn;

	/*  CUDA Handler  */
  const cudnnTensorDescriptor_t* cudnnTensorDescriptorPreP,
      *cudnnTensorDescriptorPostP;
};
}

#endif  // _TRANSPOSECONNGPU_HPP_
