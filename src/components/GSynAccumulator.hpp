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
   virtual void ioParam_channelIndices(enum ParamsIOFlag ioFlag);

   /**
    * @brief channelCoefficients: Specifies an array of coefficients for
    * the channel indices specified in the channelIndices array param.
    * If specified, channelIndices and channelCoefficients must be the same length.
    */
   virtual void ioParam_channelCoefficients(enum ParamsIOFlag ioFlag);

   /** @} */
  public:
   GSynAccumulator(char const *name, PVParams *params, Communicator *comm);

   virtual ~GSynAccumulator();

   virtual void updateBufferCPU(double simTime, double deltaTime);

  protected:
   GSynAccumulator() {}

   void initialize(char const *name, PVParams *params, Communicator *comm);

   virtual void setObjectType() override;

   virtual void initializeChannelCoefficients();

   int ioParamsFillGroup(enum ParamsIOFlag ioFlag);

   Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

#ifdef PV_USE_CUDA
   virtual void allocateUpdateKernel() override;

   virtual Response::Status copyInitialStateToGPU() override;

   virtual void updateBufferGPU(double simTime, double deltaTime) override;

   void runKernel();
#endif // PV_USE_CUDA

  protected:
   int mNumChannelIndices            = 0;
   float *mChannelIndicesParams      = nullptr; // The channel indices as provided in params
   int mNumChannelCoefficients       = 0;
   float *mChannelCoefficientsParams = nullptr; // The channel coefficients as provided in params
   std::vector<float> mChannelCoefficients;
   LayerInputBuffer *mLayerInput = nullptr;

#ifdef PV_USE_CUDA
   PVCuda::CudaBuffer *mCudaChannelCoefficients = nullptr;
#endif // PV_USE_CUDA
};

} // namespace PV

#endif // GSYNACCUMULATOR_HPP_
