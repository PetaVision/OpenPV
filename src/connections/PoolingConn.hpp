/*
 * PoolingConn.hpp
 *
 *  Created on: March 19, 2014
 *      Author: slundquist
 */

#ifndef POOLINGCONN_HPP_
#define POOLINGCONN_HPP_

#include "HyPerConn.hpp"
#include "layers/PoolingIndexLayer.hpp"
#ifdef PV_USE_CUDA
#include "cudakernels/CudaPoolingDeliverKernel.hpp"
#endif // PV_USE_CUDA

namespace PV {

class PoolingConn : public HyPerConn {

  public:
   enum AccumulateType { UNDEFINED, MAX, SUM, AVG };
   PoolingConn();
   PoolingConn(const char *name, HyPerCol *hc);
   virtual ~PoolingConn();
   PoolingIndexLayer *getPostIndexLayer() { return postIndexLayer; }
   bool needPostIndex() { return needPostIndexLayer; }
   inline AccumulateType getPoolingType() const { return poolingType; }
   static AccumulateType parseAccumulateTypeString(char const *typeString);

  protected:
   int initialize(const char *name, HyPerCol *hc);
   virtual void createWeightInitializer() override;
   virtual void createWeightNormalizer() override;
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_initializeFromCheckpointFlag(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_plasticityFlag(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_sharedWeights(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_weightInitType(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_pvpatchAccumulateType(enum ParamsIOFlag ioFlag) override;
   void ioParam_needPostIndexLayer(enum ParamsIOFlag ioFlag);
   void ioParam_postIndexLayerName(enum ParamsIOFlag ioFlag);

   /**
    * @brief PoolingConn does not have weights to write, and does not use
    * writeStep
    */
   void ioParam_writeStep(enum ParamsIOFlag ioFlag) override;
   /**
    * @brief PoolingConn does not have weights to write, and does not use
    * writeCompressedCheckpoints
    */
   void ioParam_writeCompressedCheckpoints(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief PoolingConn does not have weights to normalize, and does not use
    * normalizeMethod
    */
   void ioParam_normalizeMethod(enum ParamsIOFlag ioFlag) override;

   virtual void createDeliveryObject() override {}

   virtual void allocateWeights() override;

   virtual int
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   virtual int allocateDataStructures() override;
#ifdef PV_USE_CUDA
   virtual int allocatePostDeviceWeights() override { return PV_SUCCESS; }
   virtual int allocateDeviceWeights() override { return PV_SUCCESS; }
   virtual int initializeReceivePostKernelArgs() override { return PV_SUCCESS; }
   virtual int initializeReceivePreKernelArgs() override { return PV_SUCCESS; }
   virtual void updateDeviceWeights() override {}
   int initializeDeliverKernelArgs();
#endif // PV_USE_CUDA
   virtual int registerData(Checkpointer *checkpointer) override;

   virtual int setInitialValues() override;
   virtual int finalizeUpdate(double time, double dt) override { return PV_SUCCESS; }

   // Temporarily duplicating HyPerConn::deliver, so that HyPerConn::deliver can be overhauled
   // without breaking subclasses. -pschultz, 2017-09-08
   virtual int deliver() override;

   virtual int deliverPresynapticPerspective(PVLayerCube const *activity, int arborID);
   virtual int deliverPostsynapticPerspective(PVLayerCube const *activity, int arborID);
#ifdef PV_USE_CUDA
   virtual int deliverPresynapticPerspectiveGPU(PVLayerCube const *activity, int arborID) override;
   virtual int deliverPostsynapticPerspectiveGPU(PVLayerCube const *activity, int arborID) override;
   int deliverGPU(PVLayerCube const *activity, int arborID);
#endif // PV_USE_CUDA

   void clearGateIdxBuffer();

  private:
   int initialize_base();
   void unsetAccumulateType();
   float **thread_gateIdxBuffer;
   bool needPostIndexLayer;
   char *postIndexLayerName;
   PoolingIndexLayer *postIndexLayer;
   AccumulateType poolingType;
#ifdef PV_USE_CUDA
   PVCuda::CudaPoolingDeliverKernel *krPoolingDeliver =
         nullptr; // Cuda kernel for update state call
#endif // PV_USE_CUDA
}; // end class PoolingConn

} // end namespace PV

#endif /* CLONEKERNELCONN_HPP_ */
