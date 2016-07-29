#ifndef _HYPERCONNGPU_HPP_
#define _HYPERCONNGPU_HPP_

#include "../arch/cuda/CudaMatrix.hpp"
#include "../arch/cuda/CudaUtility.hpp"
#include "../layers/ANNLayerGPU.hpp"
#include "HyPerConn.hpp"

namespace GPULCA {

class HyPerConnGPU : PV::HyPerConn {
 public:
  HyPerConnGPU();
  HyPerConnGPU(const char* name, PV::HyPerCol* hc);
  virtual ~HyPerConnGPU();
  virtual int allocateDataStructures();
  virtual int updateState(double time, double dt);
  virtual int deliver();

  /*  get CUDA objects */
  const cudnnHandle_t& getCudnnHandle() const { return cudnnHandle; }
  const cudnnTensorDescriptor_t& getPreTensorDescriptor() const {
    return cudnnTensorDescriptorPre;
  }
  const cudnnTensorDescriptor_t& getPostTensorDescriptor() const {
    return cudnnTensorDescriptorPost;
  }
  const cusparseHandle_t& getCusparseHandle() const { return cusparseHandle; }
  const cusparseMatDescr_t& getCusparseMatDescr() const {
    return cusparseMatDescr;
  }

  /*  get private variables */
  const PVCudaWrapper<pvwdata_t>& getW() const { return W; }
  bool getIsPreGPULayerFlag() const { return isPreGPULayer; }
  bool getIsWeightSparseFlag() const { return isWeightSparse; }

 protected:
  int findCudnnAlgo();

 private:
  void initialize_base();
  int initialize();
  virtual int ioParamsFillGroup(enum PV::ParamsIOFlag ioFlag);
  virtual void ioParam_isPreGPULayer(enum PV::ParamsIOFlag ioFlag);
  virtual void ioParam_isWeightSparse(enum PV::ParamsIOFlag ioFlag);

 private:
  PVCudaWrapper<pvwdata_t>* PreNHWC, *Pre, W;
  bool isPreGPULayer, isWeightSparse;

  /*  CUDA handle */
  cudnnHandle_t cudnnHandle;
  cudnnTensorDescriptor_t cudnnTensorDescriptorPreNHWC,
      cudnnTensorDescriptorPre, cudnnTensorDescriptorPost;
  cudnnFilterDescriptor_t cudnnFilterDescriptor;
  cudnnConvolutionDescriptor_t cudnnConvolutionDescriptor;
  cudnnConvolutionFwdAlgo_t algoFwd;
  CudaVector<pvwdata_t> workspaceForward;
  size_t workspaceSizeForward;

  /*  cuSparse handle */
  cusparseHandle_t cusparseHandle;
  cusparseMatDescr_t cusparseMatDescr;
};
}

#endif  // _HYPERCONNGPU_HPP_
