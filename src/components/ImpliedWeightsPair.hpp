/*
 * ImpliedWeightsPair.hpp
 *
 *  Created on: Nov 17, 2017
 *      Author: Pete Schultz
 */

#ifndef IMPLIEDWEIGHTSPAIR_HPP_
#define IMPLIEDWEIGHTSPAIR_HPP_

#include "components/WeightsPair.hpp"

namespace PV {

// TODO: Rather than have ImpliedWeightsPair derive from WeightsPair, either have inheritance go
// the other way (since ImpliedWeightsPair turns off much of the WeightsPair behavior), or have
// both derive from an interface class.

class ImpliedWeightsPair : public WeightsPair {
  protected:
   /**
    * List of parameters needed from the ImpliedWeightsPair class
    * @name ImpliedWeightsPair Parameters
    * @{
    */

   /**
    * @brief sharedWeights: ImpliedWeightsPair does not use sharedWeights
    */
   virtual void ioParam_sharedWeights(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief writeStep: ImpliedWeightsPair does not write output
    */
   virtual void ioParam_writeStep(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief sharedWeights: ImpliedWeightsPair does not write output
    */
   virtual void ioParam_initialWriteTime(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief sharedWeights: ImpliedWeightsPair does not write output
    */
   virtual void ioParam_writeCompressedWeights(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief sharedWeights: ImpliedWeightsPair does not write checkpoints
    */
   virtual void ioParam_writeCompressedCheckpoints(enum ParamsIOFlag ioFlag) override;

   /** @} */ // end of ImpliedWeightsPair parameters

  public:
   ImpliedWeightsPair(char const *name, HyPerCol *hc);

   virtual ~ImpliedWeightsPair();

   virtual void needPre() override;
   virtual void needPost() override;

  protected:
   ImpliedWeightsPair() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual int setDescription() override;

   virtual int registerData(Checkpointer *checkpointer) override;

   virtual int readStateFromCheckpoint(Checkpointer *checkpointer) override;

   virtual void finalizeUpdate(double timestamp, double deltaTime) override;

   virtual void outputState(double timestamp) override;
};

} // namespace PV

#endif // IMPLIEDWEIGHTSPAIR_HPP_
