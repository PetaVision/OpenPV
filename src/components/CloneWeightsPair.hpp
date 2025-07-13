/*
 * CloneWeightsPair.hpp
 *
 *  Created on: Dec 3, 2017
 *      Author: Pete Schultz
 */

#ifndef CLONEWEIGHTSPAIR_HPP_
#define CLONEWEIGHTSPAIR_HPP_

#include "components/WeightsPair.hpp"

namespace PV {

class CloneWeightsPair : public WeightsPair {
  protected:
   /**
    * List of parameters needed from the CloneWeightsPair class
    * @name CloneWeightsPair Parameters
    * @{
    */

   /**
    * @brief writeStep: CloneWeightsPair never writes output, always sets writeStep to -1.
    */
   virtual void ioParam_writeStep(ParamsIOSwitch ioSwitch) override;

   /**
    * @brief writeStep: CloneWeightsPair does not checkpoint, so writeCompressedCheckpoints is
    * always set to false.
    */
   virtual void ioParam_writeCompressedCheckpoints(ParamsIOSwitch ioSwitch) override;

   /** @} */ // end of CloneWeightsPair parameters

  public:
   CloneWeightsPair(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);

   virtual ~CloneWeightsPair();

   /**
    * Synchronizes the margins of this connection's and the original connection's presynaptic
    * layers. This must be called after the two ConnectionData objects have set their pre-layer,
    * and should be called before the layers and weights enter AllocateDataStructures stage.
    */
   void synchronizeMarginsPre();

   /**
    * Synchronizes the margins of this connection's and the original connection's postsynaptic
    * layers. This must be called after the two ConnectionData objects have set their post-layer,
    * and should be called before the layers and weights enter AllocateDataStructures stage.
    */
   void synchronizeMarginsPost();

  protected:
   CloneWeightsPair() {}

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

   virtual void outputState(double timestamp) override;

  protected:
   WeightsPair *mOriginalWeightsPair = nullptr;
   ConnectionData *mOriginalConnData = nullptr;
};

} // namespace PV

#endif // CLONEWEIGHTSPAIR_HPP_
