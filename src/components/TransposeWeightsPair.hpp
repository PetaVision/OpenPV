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
    * @brief nxp: TransposeWeightsPair does not read the nxp parameter, but inherits it from the
    * originalConn's WeightsPair.
    */
   virtual void ioParam_nxp(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief nyp: TransposeWeightsPair does not read the nyp parameter, but inherits it from the
    * originalConn's WeightsPair.
    */
   virtual void ioParam_nyp(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief nfp: TransposeWeightsPair does not read the nfp parameter, but inherits it from the
    * originalConn's WeightsPair.
    */
   virtual void ioParam_nfp(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief sharedWeights: TransposeWeightsPair does not read the sharedWeights parameter,
    * but inherits it from the originalConn's WeightsPair.
    */
   virtual void ioParam_sharedWeights(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief writeStep: TransposeWeightsPair does not checkpoint, so writeCompressedCheckpoints is
    * always set to false.
    */
   virtual void ioParam_writeCompressedCheckpoints(enum ParamsIOFlag ioFlag);

   virtual void ioParam_originalConnName(enum ParamsIOFlag ioFlag);

   /** @} */ // end of TransposeWeightsPair parameters

  public:
   TransposeWeightsPair(char const *name, HyPerCol *hc);

   virtual ~TransposeWeightsPair();

   virtual void needPre() override;
   virtual void needPost() override;

   char const *getOriginalConnName() const { return mOriginalConnName; }

  protected:
   TransposeWeightsPair() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual int setDescription() override;

   int ioParamsFillGroup(enum ParamsIOFlag ioFlag);

   virtual int
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual void inferParameters();

   virtual int allocateDataStructures() override;

   virtual int registerData(Checkpointer *checkpointer) override;

   virtual void finalizeUpdate(double timestamp, double deltaTime) override;

   virtual void outputState(double timestamp) override;

  protected:
   char *mOriginalConnName = nullptr;

   WeightsPair *mOriginalWeightsPair = nullptr;
};

} // namespace PV

#endif // TRANSPOSEWEIGHTSPAIR_HPP_
