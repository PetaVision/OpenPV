#ifndef _ANNLAYERGPU_HPP_
#define _ANNLAYERGPU_HPP_

#include "../arch/cuda/CudaMatrix.hpp"
#include "../arch/cuda/CudaUtility.hpp"
#include "ANNLayer.hpp"

namespace GPULCA {

/*  this data structure may be moved the pv_types.h file. */
template <typename T>
struct PVCudaWrapper {
  DenseMatrix<T> dense;
  SparseMatrix<T> sparse;
  PVCudaWrapper() {}
  PVCudaWrapper(const MatrixInfo& params) { dense.init(params); }
  void dense2sparse() { sparse.fromDense(dense.getDeviceData()); }
  void sparse2dense() { sparse.toDense(dense.getDeviceData()); }
};

class ANNLayerGPU : public PV::ANNLayer {
 public:
  ANNLayerGPU(const char* name, PV::HyPerCol* hc);
  virtual ~ANNLayerGPU();
  PVCudaWrapper<pvdata_t>& getActivity() { return activity; }
  PVCudaWrapper<pvdata_t>& getV() { return V; }
  PVCudaWrapper<pvdata_t>& getChannel(ChannelType ch) { return GSyn.at(ch); }
  std::vector<PVCudaWrapper<pvdata_t>>& getGSyn() { return GSyn; }

  const cusparseHandle_t& getCusparseHandle() const { return cusparseHandle; }
  const cusparseMatDescr_t& getCusparseMatDescr() const {
    return cusparseMatDescr;
  }

  virtual int publish(PV::Communicator* comm, double time);

 protected:
  ANNLayerGPU();
  int initialize();
  virtual int allocateGSyn();
  virtual int allocateV();
  virtual int allocateActivity();
  virtual int resetGSynBuffers(double timef, double dt);
  virtual int setActivity();
  virtual int initializeV();
  virtual int updateState(double timef, double dt);

 private:
  void initialize_base();

 private:
  std::vector<PVCudaWrapper<pvdata_t>> GSyn;
  PVCudaWrapper<pvdata_t> activity, V;

  /*  CUDA handle */
  cusparseHandle_t cusparseHandle;
  cusparseMatDescr_t cusparseMatDescr;
};
}

#endif  // _ANNLAYERGPU_HPP_
