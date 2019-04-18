/*
 * WeightsPair.hpp
 *
 *  Created on: Nov 17, 2017
 *      Author: Pete Schultz
 */

#ifndef WEIGHTSPAIR_HPP_
#define WEIGHTSPAIR_HPP_

#include "components/ArborList.hpp"
#include "components/SharedWeights.hpp"
#include "components/WeightsPairInterface.hpp"

namespace PV {

class WeightsPair : public WeightsPairInterface {
  protected:
   /**
    * List of parameters needed from the WeightsPair class
    * @name WeightsPair Parameters
    * @{
    */

   virtual void ioParam_writeStep(enum ParamsIOFlag ioFlag);
   virtual void ioParam_initialWriteTime(enum ParamsIOFlag ioFlag);
   virtual void ioParam_writeCompressedWeights(enum ParamsIOFlag ioFlag);
   virtual void ioParam_writeCompressedCheckpoints(enum ParamsIOFlag ioFlag);

   /**
    * @brief initializeFromCheckpointFlag: If set to true, initialize using checkpoint direcgtory
    * set in HyPerCol.
    * @details Checkpoint read directory must be set in HyPerCol to initialize from checkpoint.
    */
   virtual void ioParam_initializeFromCheckpointFlag(enum ParamsIOFlag ioFlag);

   /** @} */ // end of WeightsPair parameters

  public:
   WeightsPair(char const *name, HyPerCol *hc);

   virtual ~WeightsPair();

   virtual Response::Status respond(std::shared_ptr<BaseMessage const> message) override;

   Weights *getPreWeights() { return mPreWeights; }
   Weights *getPostWeights() { return mPostWeights; }

   bool getInitializeFromCheckpointFlag() const { return mInitializeFromCheckpointFlag; }

   bool getWriteCompressedCheckpoints() const { return mWriteCompressedCheckpoints; }

   ArborList const *getArborList() const { return mArborList; }

  protected:
   WeightsPair() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual void setObjectType() override;

   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   Response::Status
   respondConnectionFinalizeUpdate(std::shared_ptr<ConnectionFinalizeUpdateMessage const> message);

   Response::Status respondConnectionOutput(std::shared_ptr<ConnectionOutputMessage const> message);

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual void createPreWeights(std::string const &weightsName) override;
   virtual void createPostWeights(std::string const &weightsName) override;

   virtual void allocatePreWeights() override;

   virtual void allocatePostWeights() override;

   virtual Response::Status registerData(Checkpointer *checkpointer) override;

   virtual Response::Status readStateFromCheckpoint(Checkpointer *checkpointer) override;

   virtual void finalizeUpdate(double timestamp, double deltaTime);

   void openOutputStateFile(Checkpointer *checkpointer);

   virtual void outputState(double timestamp);

  protected:
   double mWriteStep                = 0.0;
   double mInitialWriteTime         = 0.0;
   bool mWriteCompressedWeights     = false;
   bool mWriteCompressedCheckpoints = false;

   // If this flag is set and HyPerCol sets initializeFromCheckpointDir, load initial state from
   // the initializeFromCheckpointDir directory.
   bool mInitializeFromCheckpointFlag = false;

   ArborList *mArborList         = nullptr;
   SharedWeights *mSharedWeights = nullptr;
   double mWriteTime             = 0.0;

   CheckpointableFileStream *mOutputStateStream = nullptr; // weights file written by outputState
};

} // namespace PV

#endif // WEIGHTSPAIR_HPP_
