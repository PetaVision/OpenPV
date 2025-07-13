/*
 * SquaredGSynAccumulator.hpp
 *
 *  Created on: Sep 11, 2018
 *      Author: Pete Schultz
 */

#ifndef SQUAREDGSYNACCUMULATOR_HPP_
#define SQUAREDGSYNACCUMULATOR_HPP_

#include "components/SingleChannelGSynAccumulator.hpp"

#include "components/LayerInputBuffer.hpp"

namespace PV {

/**
 * A component to contain the internal state (membrane potential) of a HyPerLayer.
 */
class SquaredGSynAccumulator : public SingleChannelGSynAccumulator {
  public:
   SquaredGSynAccumulator(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);

   virtual ~SquaredGSynAccumulator();

   virtual void updateBufferCPU(double simTime, double deltaTime) override;

  protected:
   SquaredGSynAccumulator() {}

   void initialize(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);

   virtual void setObjectType() override;

   void initializeChannelCoefficients() override;
};

} // namespace PV

#endif // SQUAREDGSYNACCUMULATOR_HPP_
