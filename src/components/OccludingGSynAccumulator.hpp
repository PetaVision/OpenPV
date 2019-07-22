/*
 * OccludingGSynAccumulator.hpp
 *
 *  Created on: Jul 18, 2019
 *      Author: Jacob Springer
 */

#ifndef OCCLUDINGGSYNACCUMULATOR_HPP_
#define OCCLUDINGGSYNACCUMULATOR_HPP_

#include "components/RestrictedBuffer.hpp"
#include "components/LayerInputBuffer.hpp"
#include "components/GSynAccumulator.hpp"

namespace PV {

/**
 * A component to contain the internal state (membrane potential) of a HyPerLayer.
 */
class OccludingGSynAccumulator : public GSynAccumulator {
  protected:

   void ioParam_opaqueMagnitude(enum ParamsIOFlag ioFlag);

   /** @} */
  public:
   OccludingGSynAccumulator(char const *name, PVParams *params, Communicator const *comm);

   virtual ~OccludingGSynAccumulator();

   virtual void updateBufferCPU(double simTime, double deltaTime) override;

   float const* retrieveContribData();
  protected:
   OccludingGSynAccumulator() {}

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual void setObjectType() override;

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual Response::Status allocateDataStructures() override;


#ifdef PV_USE_CUDA
   virtual void allocateUpdateKernel() override;

   virtual Response::Status copyInitialStateToGPU() override;

   virtual void updateBufferGPU(double simTime, double deltaTime) override;

   void runKernel();
#endif // PV_USE_CUDA

  protected:
   int mNumChannels              = 0;
   LayerInputBuffer *mLayerInput = nullptr;
   std::vector<float> mContribData;
   float mOpaqueMagnitude        = 1.0;

#ifdef PV_USE_CUDA
   PVCuda::CudaBuffer *mCudaContribData = nullptr;
#endif
};

} // namespace PV

#endif // OCCLUDINGGSYNACCUMULATOR_HPP_
