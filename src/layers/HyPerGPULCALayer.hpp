#ifndef _HYPERGPULCALAYER_HPP_
#define _HYPERGPULCALAYER_HPP_

#include "../arch/cuda/CudaMatrix.hpp"
#include "../arch/cuda/CudaUtility.hpp"
#include "ANNLayerGPU.hpp"

namespace GPULCA {

class HyPerGPULCALayer : public ANNLayerGPU {
 public:
  HyPerGPULCALayer();
  HyPerGPULCALayer(const char* name, PV::HyPerCol* hc);
  ~HyPerGPULCALayer();

  virtual int updateState(double time, double dt);

  virtual pvdata_t getChannelTimeConst(enum ChannelType channel_type) {
    return timeConstantTau;
  };

  const cublasHandle_t& getCublasHandle() const { return cublasHandle; }

  /*  private functions  */
 private:
  int init();
  int initialize_base();
  virtual int ioParamsFillGroup(enum PV::ParamsIOFlag ioFlag);
  virtual void ioParam_timeConstantTau(enum PV::ParamsIOFlag ioFlag);

 private:
  PVCudaWrapper<pvdata_t> UDot;
  SparseMatrix<pvdata_t> identity;
  pvdata_t timeConstantTau;

  /*  CUDA handle */
  cublasHandle_t cublasHandle;
};
}

#endif  //_HYPERGPULCA_HPP_
