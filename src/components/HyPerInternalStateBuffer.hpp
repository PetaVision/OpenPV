/*
 * HyPerInternalStateBuffer.hpp
 *
 *  Created on: Oct 12, 2018 from the original HyPerLayer
 *      Author: Pete Schultz
 */

#ifndef HYPERINTERNALSTATEBUFFER_HPP_
#define HYPERINTERNALSTATEBUFFER_HPP_

#include "components/InternalStateBuffer.hpp"

#include "components/LayerInputBuffer.hpp"

namespace PV {

/**
 * A component to contain the internal state (membrane potential) of a HyPerLayer.
 */
class HyPerInternalStateBuffer : public InternalStateBuffer {
  protected:
   /**
    * List of parameters needed from the HyPerInternalStateBuffer class
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
   HyPerInternalStateBuffer(char const *name, PVParams *params, Communicator const *comm);

   virtual ~HyPerInternalStateBuffer();

  protected:
   HyPerInternalStateBuffer() {}

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual void setObjectType() override;

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual Response::Status allocateDataStructures() override;

   /**
    * Computes the buffer as excitatory input minus inhibitory input from the LayerInput buffer.
    * The previous internal state has no effect on the new internal state.
    */
   virtual void updateBufferCPU(double simTime, double deltaTime) override;

#ifdef PV_USE_CUDA
   virtual void allocateUpdateKernel() override;

   virtual Response::Status copyInitialStateToGPU() override;

   virtual void updateBufferGPU(double simTime, double deltaTime) override;

   void runKernel();
#endif // PV_USE_CUDA

  protected:
   // If the ChannelIndices and ChannelCoefficients params are absent, we will use
   // The ChannelIndices and ChannelCoefficients arrays as given in params might be different from
   // what actually gets used. Most significantly, if they are absent from the params file, the
   // arrays will be null and the number of entries will be zero, but what gets used are the default
   // two channels and the coefficients +1 for excitatory and -1 for inhibitory.
   // We keep these as separate data members so that the PV-generated params file is the same as
   // the input.

   int mNumChannelIndicesParams = 0;  // The number of channel indices & coefficients in params
   int *mChannelIndicesParams;        // The channel indices in params
   float *mChannelCoefficientsParams; // The channel coefficients in params

   // It may happen that the LayerInputBuffer does not have as many channels as are specified in
   // params. For example, params might specify both excitatory and inhibitory coefficients but
   // the layer input buffer has only an excitatory connection. We trim the indices and
   // coefficients to only the values needed.
   int mNumChannelIndices = 0;              // The number of channel indices in practice
   std::vector<int> mChannelIndices;        // The channel indices in practice
   std::vector<float> mChannelCoefficients; // The channel coefficients in practice

   LayerInputBuffer *mGSyn = nullptr;

#ifdef PV_USE_CUDA
   PVCuda::CudaBuffer *mCudaChannelIndices      = nullptr;
   PVCuda::CudaBuffer *mCudaChannelCoefficients = nullptr;
#endif // PV_USE_CUDA
};

} // namespace PV

#endif // HYPERINTERNALSTATEBUFFER_HPP_
