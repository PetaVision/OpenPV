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

class PoolingConn: public HyPerConn {

public:
   enum AccumulateType {UNDEFINED, MAX, SUM, AVG};
   PoolingConn();
   PoolingConn(const char * name, HyPerCol * hc);
   virtual ~PoolingConn();
   virtual int communicateInitInfo();
   virtual int allocateDataStructures();
   virtual int checkpointRead(const char * cpDir, double* timef) override { return PV_SUCCESS; }
   virtual int checkpointWrite(const char * cpDir) override { return PV_SUCCESS; }
   virtual float minWeight(int arborId = 0);
   virtual float maxWeight(int arborId = 0);
   virtual int finalizeUpdate(double time, double dt) override { return PV_SUCCESS; }
   PoolingIndexLayer* getPostIndexLayer(){return postIndexLayer;}
   bool needPostIndex(){return needPostIndexLayer;}
   inline AccumulateType getPoolingType() const { return poolingType; }
   static AccumulateType parseAccumulateTypeString(char const * typeString);

protected:
   int initialize(const char * name, HyPerCol * hc, InitWeights * weightInitializer, NormalizeBase * weightNormalizer);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   void ioParam_plasticityFlag(enum ParamsIOFlag ioFlag);
   void ioParam_weightInitType(enum ParamsIOFlag ioFlag);
   void ioParam_pvpatchAccumulateType(enum ParamsIOFlag ioFlag);
   void ioParam_needPostIndexLayer(enum ParamsIOFlag ioFlag);
   void ioParam_postIndexLayerName(enum ParamsIOFlag ioFlag);

   /**
    * @brief PoolingConn does not have weights to write, and does not use writeStep
    */
   void ioParam_writeStep(enum ParamsIOFlag ioFlag);
   /**
    * @brief PoolingConn does not have weights to write, and does not use writeCompressedCheckpoints
    */
   void ioParam_writeCompressedCheckpoints(enum ParamsIOFlag ioFlag);

   /**
    * @brief PoolingConn does not have weights to normalize, and does not use normalizeMethod
    */
   void ioParam_normalizeMethod(enum ParamsIOFlag ioFlag);
#ifdef PV_USE_CUDA
   virtual int allocateDeviceWeights() override { return PV_SUCCESS; }
   virtual int initializeReceivePostKernelArgs() override { return PV_SUCCESS; }
   virtual int initializeReceivePreKernelArgs() override { return PV_SUCCESS; }
   virtual void updateDeviceWeights() override {}
   int initializeDeliverKernelArgs();
#endif // PV_USE_CUDA
   virtual int setInitialValues() override;
   virtual int constructWeights();

   virtual int deliverPresynapticPerspective(PVLayerCube const * activity, int arborID) override;
   virtual int deliverPostsynapticPerspective(PVLayerCube const * activity, int arborID) override;
#ifdef PV_USE_CUDA
   virtual int deliverPresynapticPerspectiveGPU(PVLayerCube const * activity, int arborID) override;
   virtual int deliverPostsynapticPerspectiveGPU(PVLayerCube const * activity, int arborID) override;
   int deliverGPU(PVLayerCube const * activity, int arborID);
#endif // PV_USE_CUDA

   void clearGateIdxBuffer();

private:
   int initialize_base();
   void unsetAccumulateType();
   pvdata_t ** thread_gateIdxBuffer;
   bool needPostIndexLayer;
   char* postIndexLayerName;
   PoolingIndexLayer* postIndexLayer;
   AccumulateType poolingType;
#ifdef PV_USE_CUDA
   PVCuda::CudaPoolingDeliverKernel* krPoolingDeliver = nullptr; // Cuda kernel for update state call
#endif // PV_USE_CUDA
}; // end class PoolingConn

}  // end namespace PV

#endif /* CLONEKERNELCONN_HPP_ */
