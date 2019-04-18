/*
 * CopyWeightsPair.hpp
 *
 *  Created on: Dec 15, 2017
 *      Author: Pete Schultz
 */

#ifndef COPYWEIGHTSPAIR_HPP_
#define COPYWEIGHTSPAIR_HPP_

#include "components/WeightsPair.hpp"
#include "connections/HyPerConn.hpp"

namespace PV {

/**
 * A derived class of WeightsPair that copies weights from another connection, specified
 * in the originalConnName parameter. The nxp, nyp, nfp, and sharedWeights parameters
 * are copied from the original connection, not read from the params file.
 * The weights are copied from the original using the copy() method (which is called
 * by CopyConn::initializeState and CopyUpdater::updateState).
 *
 */
class CopyWeightsPair : public WeightsPair {
  public:
   CopyWeightsPair(char const *name, HyPerCol *hc);

   virtual ~CopyWeightsPair();

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

   /**
    * Copies the weights from the original weights pair.
    */
   void copy();

   WeightsPair const *getOriginalWeightsPair() const { return mOriginalWeightsPair; }

  protected:
   CopyWeightsPair() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual void setObjectType() override;

   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual void createPreWeights(std::string const &weightsName) override;
   virtual void createPostWeights(std::string const &weightsName) override;

  protected:
   HyPerConn *mOriginalConn          = nullptr;
   WeightsPair *mOriginalWeightsPair = nullptr;
};

} // namespace PV

#endif // COPYWEIGHTSPAIR_HPP_
