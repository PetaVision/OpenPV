/*
 * SingleChannelGSynAccumulator.hpp
 *
 *  Created on: Sep 11, 2018
 *      Author: Pete Schultz
 */

#ifndef SINGLECHANNELGSYNACCUMULATOR_HPP_
#define SINGLECHANNELGSYNACCUMULATOR_HPP_

#include "components/GSynAccumulator.hpp"

#include "components/LayerInputBuffer.hpp"

namespace PV {

/**
 * A component to contain the internal state (membrane potential) of a HyPerLayer.
 */
class SingleChannelGSynAccumulator : public GSynAccumulator {
  protected:
   /**
    * List of parameters needed from the SingleChannelGSynAccumulator class
    * @name HyPerLayer Parameters
    * @{
    */

   /**
    * @brief channelIndices: SingleChannelGSynAccumulator does not use channelIndices.
    * channel coefficients will be specified.
    */
   virtual void ioParam_channelIndices(enum ParamsIOFlag ioFlag);

   /**
    * @brief channelIndices: SingleChannelGSynAccumulator does not use channelCoefficients.
    */
   virtual void ioParam_channelCoefficients(enum ParamsIOFlag ioFlag);

   /** @} */
  public:
   SingleChannelGSynAccumulator(char const *name, PVParams *params, Communicator *comm);

   virtual ~SingleChannelGSynAccumulator();

   virtual void updateBufferCPU(double simTime, double deltaTime);

  protected:
   SingleChannelGSynAccumulator() {}

   void initialize(char const *name, PVParams *params, Communicator *comm);

   virtual void setObjectType() override;

   void initializeChannelCoefficients() override;

   Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

  protected:
};

} // namespace PV

#endif // SINGLECHANNELGSYNACCUMULATOR_HPP_
