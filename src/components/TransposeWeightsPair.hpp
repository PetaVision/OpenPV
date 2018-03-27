/*
 * TransposeWeightsPair.hpp
 *
 *  Created on: Dec 3, 2017
 *      Author: Pete Schultz
 */

#ifndef TRANSPOSEWEIGHTSPAIR_HPP_
#define TRANSPOSEWEIGHTSPAIR_HPP_

#include "components/WeightsPair.hpp"
#include "connections/HyPerConn.hpp"

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
   virtual void ioParam_writeCompressedCheckpoints(enum ParamsIOFlag ioFlag) override;

   /** @} */ // end of TransposeWeightsPair parameters

  public:
   TransposeWeightsPair(char const *name, HyPerCol *hc);

   virtual ~TransposeWeightsPair();

  protected:
   TransposeWeightsPair() {}

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

#endif // TRANSPOSEWEIGHTSPAIR_HPP_
