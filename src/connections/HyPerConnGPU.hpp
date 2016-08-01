#ifndef _HYPERCONNGPU_HPP_
#define _HYPERCONNGPU_HPP_

#include "../arch/cuda/CudaMatrix.hpp"
#include "../arch/cuda/CudaUtility.hpp"
#include "../layers/ANNLayerGPU.hpp"
#include "HyPerConn.hpp"

namespace GPULCA {

class HyPerConnGPU : public PV::HyPerConn {
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
  const std::vector<PVCudaWrapper<pvwdata_t>>& getW() const { return W; }
  std::vector<PVCudaWrapper<pvwdata_t>>& getW() { return W; }
	const std::vector<PVCudaWrapper<pvwdata_t>>& getWT() const { return WT; }
  std::vector<PVCudaWrapper<pvwdata_t>>& getWT() { return WT; }
  const PVCudaWrapper<pvwdata_t>& getW(int arborId) const {
    return W.at(arborId);
  }
  PVCudaWrapper<pvwdata_t>& getW(int arborId) { return W.at(arborId); }
  bool getIsPreGPULayerFlag() const { return isPreGPULayer; }
  bool getIsWeightSparseFlag() const { return isWeightSparse; }

 protected:
  virtual int findCudnnAlgo();
  virtual int update_dW(int arborId);
  virtual int updateWeights(int arborId = 0);

 private:
  void initialize_base();
  int initialize();
  virtual int ioParamsFillGroup(enum PV::ParamsIOFlag ioFlag);
  virtual void ioParam_isPreGPULayer(enum PV::ParamsIOFlag ioFlag);
  virtual void ioParam_isWeightSparse(enum PV::ParamsIOFlag ioFlag);
	int computeTransposeMap();
	int transposeWeight();

 private:
  PVCudaWrapper<pvwdata_t>* PreNHWC, *Pre;
  std::vector<PVCudaWrapper<pvwdata_t>> W, WT;  // W: KCHW
                                                // WT:CKHW
  bool isPreGPULayer, isWeightSparse;
	DenseMatrix<int> map;

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
