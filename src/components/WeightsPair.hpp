/*
 * WeightsPair.hpp
 *
 *  Created on: Nov 17, 2017
 *      Author: Pete Schultz
 */

#ifndef WEIGHTSPAIR_HPP_
#define WEIGHTSPAIR_HPP_

#include "columns/BaseObject.hpp"
#include "components/ConnectionData.hpp"
#include "components/PostWeights.hpp"
#include "components/Weights.hpp"

namespace PV {

class WeightsPair : public BaseObject {
  protected:
   /**
    * List of parameters needed from the WeightsPair class
    * @name WeightsPair Parameters
    * @{
    */

   /**
    * @brief nxp: Specifies the x patch size
    * @details If one pre to many post, nxp restricted to many * an odd number
    * If many pre to one post or one pre to one post, nxp restricted to an odd number
    */
   virtual void ioParam_nxp(enum ParamsIOFlag ioFlag);

   /**
    * @brief nyp: Specifies the y patch size
    * @details If one pre to many post, nyp restricted to many * an odd number
    * If many pre to one post or one pre to one post, nyp restricted to an odd number
    */
   virtual void ioParam_nyp(enum ParamsIOFlag ioFlag);

   /**
    * @brief nfp: Specifies the post feature patch size
    */
   virtual void ioParam_nfp(enum ParamsIOFlag ioFlag);

   /**
    * @brief sharedWeights: Defines if the weights use shared weights
    */
   virtual void ioParam_sharedWeights(enum ParamsIOFlag ioFlag);
   virtual void ioParam_writeStep(enum ParamsIOFlag ioFlag);
   virtual void ioParam_initialWriteTime(enum ParamsIOFlag ioFlag);
   virtual void ioParam_writeCompressedWeights(enum ParamsIOFlag ioFlag);
   virtual void ioParam_writeCompressedCheckpoints(enum ParamsIOFlag ioFlag);

   /** @} */ // end of WeightsPair parameters

  public:
   WeightsPair(char const *name, HyPerCol *hc);

   virtual ~WeightsPair();

   int respond(std::shared_ptr<BaseMessage const> message);

   int getPatchSizeX() const { return mPatchSizeX; }
   int getPatchSizeY() const { return mPatchSizeY; }
   int getPatchSizeF() const { return mPatchSizeF; }
   int getSharedWeights() const { return mSharedWeights; }

   virtual void needPre();
   virtual void needPost();

   Weights *getPreWeights() { return mPreWeights; }
   Weights *getPostWeights() { return mPostWeights; }

   bool getWriteCompressedCheckpoints() const { return mWriteCompressedCheckpoints; }

  protected:
   WeightsPair() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual int setDescription() override;

   int ioParamsFillGroup(enum ParamsIOFlag ioFlag);

   int respondConnectionOutput(std::shared_ptr<ConnectionOutputMessage const> message);

   virtual int
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual int allocateDataStructures() override;

   virtual void allocatePreWeights();

   virtual void allocatePostWeights();

   virtual int registerData(Checkpointer *checkpointer) override;

   virtual int readStateFromCheckpoint(Checkpointer *checkpointer) override;

   void openOutputStateFile(Checkpointer *checkpointer);

   virtual void outputState(double timestamp);

  protected:
   int mPatchSizeX          = 0;
   int mPatchSizeY          = 0;
   int mPatchSizeF          = -1;
   bool mSharedWeights      = false;
   double mWriteStep        = 0.0;
   double mInitialWriteTime = 0.0;

   bool mWriteCompressedWeights     = false;
   bool mWriteCompressedCheckpoints = false;

   ConnectionData *mConnectionData = nullptr;

   Weights *mPreWeights  = nullptr;
   Weights *mPostWeights = nullptr;
   double mWriteTime     = 0.0;

   CheckpointableFileStream *mOutputStateStream = nullptr; // weights file written by outputState

   bool mWarnDefaultNfp = true;
   // Whether to print a warning if the default nfp is used.
   // Derived classes can set to false if no warning is necessary.
};

} // namespace PV

#endif // WEIGHTSPAIR_HPP_
