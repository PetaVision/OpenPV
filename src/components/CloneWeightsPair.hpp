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
    * @brief nxp: CloneWeightsPair does not read the nxp parameter, but inherits it from the
    * originalConn's WeightsPair.
    */
   virtual void ioParam_nxp(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief nyp: CloneWeightsPair does not read the nyp parameter, but inherits it from the
    * originalConn's WeightsPair.
    */
   virtual void ioParam_nyp(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief nfp: CloneWeightsPair does not read the nfp parameter, but inherits it from the
    * originalConn's WeightsPair.
    */
   virtual void ioParam_nfp(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief sharedWeights: CloneWeightsPair does not read the sharedWeights parameter,
    * but inherits it from the originalConn's WeightsPair.
    */
   virtual void ioParam_sharedWeights(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief writeStep: CloneWeightsPair never writes output, always sets writeStep to -1.
    */
   virtual void ioParam_writeStep(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief writeStep: CloneWeightsPair does not checkpoint, so writeCompressedCheckpoints is
    * always set to false.
    */
   virtual void ioParam_writeCompressedCheckpoints(enum ParamsIOFlag ioFlag) override;

   virtual void ioParam_originalConnName(enum ParamsIOFlag ioFlag);

   /** @} */ // end of CloneWeightsPair parameters

  public:
   CloneWeightsPair(char const *name, HyPerCol *hc);

   virtual ~CloneWeightsPair();

   virtual void needPre() override;
   virtual void needPost() override;

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

   char const *getOriginalConnName() const { return mOriginalConnName; }

  protected:
   CloneWeightsPair() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual int setDescription() override;

   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual int
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual void copyParameters();

   virtual int allocateDataStructures() override;

   virtual int registerData(Checkpointer *checkpointer) override;

   virtual void finalizeUpdate(double timestamp, double deltaTime) override;

   virtual void outputState(double timestamp) override;

  protected:
   char *mOriginalConnName           = nullptr;
   HyPerConn *mOriginalConn          = nullptr;
   WeightsPair *mOriginalWeightsPair = nullptr;
};

} // namespace PV

#endif // CLONEWEIGHTSPAIR_HPP_
