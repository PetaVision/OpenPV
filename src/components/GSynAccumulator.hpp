/*
 * GSynAccumulator.hpp
 *
 *  Created on: Nov 14, 2018
 *      Author: Pete Schultz
 */

#ifndef GSYNACCUMULATOR_HPP_
#define GSYNACCUMULATOR_HPP_

#include "components/RestrictedBuffer.hpp"

#include "components/LayerInputBuffer.hpp"

namespace PV {

/**
 * A component to contain the internal state (membrane potential) of a HyPerLayer.
 */
class GSynAccumulator : public RestrictedBuffer {
  protected:
   /**
    * List of parameters needed from the GSynAccumulator class
    * @name HyPerLayer Parameters
    * @{
    */

   /**
    * @brief channelIndices: Specifies an array of channel indices for which
    * channel coefficients will be specified.
    */
   virtual void ioParam_channelIndices(ParamsIOSwitch ioSwitch);

   /**
    * @brief channelCoefficients: Specifies an array of coefficients for
    * the channel indices specified in the channelIndices array param.
    * If specified, channelIndices and channelCoefficients must be the same length.
    */
   virtual void ioParam_channelCoefficients(ParamsIOSwitch ioSwitch);

   /** @} */
  public:
   GSynAccumulator(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);

   virtual ~GSynAccumulator();

   virtual void updateBufferCPU(double simTime, double deltaTime) override;

  protected:
   GSynAccumulator() {}

   void initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);

   virtual void setObjectType() override;

   virtual void initializeChannelCoefficients();

   virtual int ioParamsFillGroup(ParamsIOSwitch ioSwitch) override;

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
   std::vector<int> mChannelIndicesParams; // The channel indices as provided in params
   std::vector<float> mChannelCoefficientsParams; // The channel coefficients as provided in params
   std::vector<float> mChannelCoefficients;
   LayerInputBuffer *mLayerInput = nullptr;

   int mNumInputChannels = 0;
// The smaller of the number of channel coefficients and mLayerInput's NumChannels.
// Set in the Allocate stage.

#ifdef PV_USE_CUDA
   PVCuda::CudaBuffer *mCudaChannelCoefficients = nullptr;
#endif // PV_USE_CUDA
};

} // namespace PV

#endif // GSYNACCUMULATOR_HPP_
