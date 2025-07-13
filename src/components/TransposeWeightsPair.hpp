/*
 * TransposeWeightsPair.hpp
 *
 *  Created on: Dec 3, 2017
 *      Author: Pete Schultz
 */

#ifndef TRANSPOSEWEIGHTSPAIR_HPP_
#define TRANSPOSEWEIGHTSPAIR_HPP_

#include "components/WeightsPair.hpp"

namespace PV {

class TransposeWeightsPair : public WeightsPair {
  protected:
   /**
    * List of parameters needed from the TransposeWeightsPair class
    * @name TransposeWeightsPair Parameters
    * @{
    */

   /**
    * @brief writeStep: TransposeWeightsPair does not checkpoint, so writeCompressedCheckpoints is
    * always set to false.
    */
   virtual void ioParam_writeCompressedCheckpoints(ParamsIOSwitch ioSwitch) override;

   /** @} */ // end of TransposeWeightsPair parameters

  public:
   TransposeWeightsPair(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);

   virtual ~TransposeWeightsPair();

  protected:
   TransposeWeightsPair() {}

   void initialize(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);

   virtual void setObjectType() override;

   virtual int ioParamsFillGroup(ParamsIOSwitch ioSwitch) override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual void createPreWeights(std::string const &weightsName) override;
   virtual void createPostWeights(std::string const &weightsName) override;

   virtual Response::Status allocateDataStructures() override;

   virtual Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;

   virtual void finalizeUpdate(double timestamp, double deltaTime) override;

  protected:
   WeightsPair *mOriginalWeightsPair = nullptr;
};

} // namespace PV

#endif // TRANSPOSEWEIGHTSPAIR_HPP_
