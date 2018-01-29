/*
 * CloneWeightsPair.hpp
 *
 *  Created on: Dec 3, 2017
 *      Author: Pete Schultz
 */

#ifndef CLONEWEIGHTSPAIR_HPP_
#define CLONEWEIGHTSPAIR_HPP_

#include "components/WeightsPair.hpp"
#include "connections/HyPerConn.hpp"

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
   virtual void ioParam_writeStep(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief writeStep: CloneWeightsPair does not checkpoint, so writeCompressedCheckpoints is
    * always set to false.
    */
   virtual void ioParam_writeCompressedCheckpoints(enum ParamsIOFlag ioFlag) override;

   /** @} */ // end of CloneWeightsPair parameters

  public:
   CloneWeightsPair(char const *name, HyPerCol *hc);

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

   int initialize(char const *name, HyPerCol *hc);

   virtual void setObjectType() override;

   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual void createPreWeights(std::string const &weightsName) override;
   virtual void createPostWeights(std::string const &weightsName) override;

   virtual Response::Status allocateDataStructures() override;

   virtual Response::Status registerData(Checkpointer *checkpointer) override;

   virtual void finalizeUpdate(double timestamp, double deltaTime) override;

   virtual void outputState(double timestamp) override;

  protected:
   HyPerConn *mOriginalConn          = nullptr;
   WeightsPair *mOriginalWeightsPair = nullptr;
};

} // namespace PV

#endif // CLONEWEIGHTSPAIR_HPP_
