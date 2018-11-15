/*
 * PtwiseQuotientGSynAccumulator.hpp
 *
 *  Created on: Sep 11, 2018
 *      Author: Pete Schultz
 */

#ifndef PTWISEQUOTIENTGSYNACCUMULATOR_HPP_
#define PTWISEQUOTIENTGSYNACCUMULATOR_HPP_

#include "components/GSynAccumulator.hpp"

#include "components/LayerInputBuffer.hpp"

namespace PV {

/**
 * A component to contain the internal state (membrane potential) of a HyPerLayer.
 */
class PtwiseQuotientGSynAccumulator : public GSynAccumulator {
  protected:
   /**
    * List of parameters needed from the PtwiseQuotientGSynAccumulator class
    * @name HyPerLayer Parameters
    * @{
    */

   /**
    * @brief channelIndices: PtwiseQuotientGSynAccumulator does not use channelIndices.
    * channel coefficients will be specified.
    */
   virtual void ioParam_channelIndices(enum ParamsIOFlag ioFlag);

   /**
    * @brief channelIndices: PtwiseQuotientGSynAccumulator does not use channelCoefficients.
    */
   virtual void ioParam_channelCoefficients(enum ParamsIOFlag ioFlag);

   /** @} */
  public:
   PtwiseQuotientGSynAccumulator(char const *name, PVParams *params, Communicator *comm);

   virtual ~PtwiseQuotientGSynAccumulator();

   virtual void updateBufferCPU(double simTime, double deltaTime);

  protected:
   PtwiseQuotientGSynAccumulator() {}

   void initialize(char const *name, PVParams *params, Communicator *comm);

   virtual void setObjectType() override;

   Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

  protected:
};

} // namespace PV

#endif // PTWISEQUOTIENTGSYNACCUMULATOR_HPP_
