/*
 * TransposeConn.hpp
 *
 *  Created on: May 16, 2011
 *      Author: peteschultz
 */

#ifndef TRANSPOSECONN_HPP_
#define TRANSPOSECONN_HPP_

#include "HyPerConn.hpp"

namespace PV {

class TransposeConn : public HyPerConn {
  public:
   TransposeConn();
   TransposeConn(const char *name, HyPerCol *hc);
   virtual ~TransposeConn();
   virtual int
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   virtual int allocateDataStructures() override;
   inline HyPerConn *getOriginalConn() { return originalConn; }

   virtual bool needUpdate(double timed, double dt) override;
   virtual int updateState(double time, double dt) override;
   virtual double computeNewWeightUpdateTime(double time, double currentUpdateTime) override;
   virtual int finalizeUpdate(double time, double dt) override;

#ifdef PV_USE_CUDA
   // If this layer needs to allocate device weights, set orig conn's alloc post
   // weights
   virtual void setAllocDeviceWeights() override { originalConn->setAllocPostDeviceWeights(); }
   // Vice versa
   virtual void setAllocPostDeviceWeights() override { originalConn->setAllocDeviceWeights(); }
#endif // PV_USE_CUDA

#ifdef PV_USE_CUDA
   virtual PVCuda::CudaBuffer *getDeviceWData() override {
      return originalConn->postConn->getDeviceWData();
   }
#endif

#if defined(PV_USE_CUDA) && defined(PV_USE_CUDNN)
   virtual PVCuda::CudaBuffer *getCudnnWData() override {
      return originalConn->postConn->getCudnnWData();
   }
#endif

  protected:
#ifdef PV_USE_CUDA
   virtual int allocatePostDeviceWeights() override;
   virtual int allocateDeviceWeights() override;
#endif

   int initialize_base();
   int initialize(const char *name, HyPerCol *hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_sharedWeights(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_weightInitType(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_initializeFromCheckpointFlag(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_numAxonalArbors(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_plasticityFlag(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_triggerLayerName(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_combine_dW_with_W_flag(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_nxp(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_nyp(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_nfp(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_dWMax(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_weightUpdatePeriod(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_initialWeightUpdateTime(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_shrinkPatches(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_normalizeMethod(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_originalConnName(enum ParamsIOFlag ioFlag);
   virtual int setPatchSize() override;
   virtual int setNeededRNGSeeds() { return 0; }
   virtual int registerData(Checkpointer *checkpointer) override;
   virtual int setInitialValues() override;
   virtual PVPatch ***initializeWeights(PVPatch ***arbors, float **dataStart) override;
   virtual int constructWeights() override;
   virtual int allocatePostConn() override;

  private:
   int deleteWeights();

   // Member variables
  protected:
   char *originalConnName;
   HyPerConn *originalConn;
}; // end class TransposeConn

} // end namespace PV

#endif /* TRANSPOSECONN_HPP_ */
