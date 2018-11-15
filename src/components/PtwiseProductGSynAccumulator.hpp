/*
 * PtwiseProductGSynAccumulator.hpp
 *
 *  Created on: Sep 11, 2018
 *      Author: Pete Schultz
 */

#ifndef PTWISEPRODUCTGSYNACCUMULATOR_HPP_
#define PTWISEPRODUCTGSYNACCUMULATOR_HPP_

#include "components/GSynAccumulator.hpp"

#include "components/LayerInputBuffer.hpp"

namespace PV {

/**
 * A component to contain the internal state (membrane potential) of a HyPerLayer.
 */
class PtwiseProductGSynAccumulator : public GSynAccumulator {
  protected:
   /**
    * List of parameters needed from the PtwiseProductGSynAccumulator class
    * @name HyPerLayer Parameters
    * @{
    */

   /**
    * @brief channelIndices: PtwiseProductGSynAccumulator does not use channelIndices.
    * channel coefficients will be specified.
    */
   virtual void ioParam_channelIndices(enum ParamsIOFlag ioFlag);

   /**
    * @brief channelIndices: PtwiseProductGSynAccumulator does not use channelCoefficients.
    */
   virtual void ioParam_channelCoefficients(enum ParamsIOFlag ioFlag);

   /** @} */
  public:
   PtwiseProductGSynAccumulator(char const *name, PVParams *params, Communicator *comm);

   virtual ~PtwiseProductGSynAccumulator();

   virtual void updateBufferCPU(double simTime, double deltaTime);

  protected:
   PtwiseProductGSynAccumulator() {}

   void initialize(char const *name, PVParams *params, Communicator *comm);

   virtual void setObjectType() override;

   Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

  protected:
};

} // namespace PV

#endif // PTWISEPRODUCTGSYNACCUMULATOR_HPP_
