/*
 * WeightsPair.hpp
 *
 *  Created on: Nov 17, 2017
 *      Author: Pete Schultz
 */

#ifndef WEIGHTSPAIR_HPP_
#define WEIGHTSPAIR_HPP_

#include "components/ArborList.hpp"
#include "components/WeightsPairInterface.hpp"
#include "io/WeightsFile.hpp"
#include "structures/Weights.hpp"
#include <memory>

namespace PV {

class WeightsPair : public WeightsPairInterface {
  protected:
   /**
    * List of parameters needed from the WeightsPair class
    * @name WeightsPair Parameters
    * @{
    */

   virtual void ioParam_writeStep(ParamsIOSwitch ioSwitch);
   virtual void ioParam_initialWriteTime(ParamsIOSwitch ioSwitch);
   virtual void ioParam_writeCompressedWeights(ParamsIOSwitch ioSwitch);
   virtual void ioParam_writeCompressedCheckpoints(ParamsIOSwitch ioSwitch);

   /** @} */ // end of WeightsPair parameters

  public:
   WeightsPair(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);

   virtual ~WeightsPair();

   Weights *getPreWeights() { return mPreWeights; }
   Weights *getPostWeights() { return mPostWeights; }

   // param accessor methods
   double getWriteStep() const { return mWriteStep; }
   double getInitialWriteTime() const { return mInitialWriteTime; }
   bool getWriteCompressedWeights() const { return mWriteCompressedWeights; }
   bool getWriteCompressedCheckpoints() const { return mWriteCompressedCheckpoints; }

   ArborList const *getArborList() const { return mArborList; }

  protected:
   WeightsPair() {}

   void initialize(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);

   virtual void setObjectType() override;

   virtual void initMessageActionMap() override;

   virtual int ioParamsFillGroup(ParamsIOSwitch ioSwitch) override;

   Response::Status
   respondConnectionFinalizeUpdate(std::shared_ptr<ConnectionFinalizeUpdateMessage const> message);

   Response::Status respondConnectionOutput(std::shared_ptr<ConnectionOutputMessage const> message);

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual void createPreWeights(std::string const &weightsName) override;
   virtual void createPostWeights(std::string const &weightsName) override;

   virtual void allocatePreWeights() override;

   virtual void allocatePostWeights() override;

   virtual Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;

   virtual Response::Status readStateFromCheckpoint(Checkpointer *checkpointer) override;

   virtual void finalizeUpdate(double timestamp, double deltaTime);

   void openOutputStateFile(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message);

   virtual void outputState(double timestamp);

  protected:
   double mWriteStep                = 0.0;
   double mInitialWriteTime         = 0.0;
   bool mWriteCompressedWeights     = false;
   bool mWriteCompressedCheckpoints = false;

   ArborList *mArborList         = nullptr;
   double mWriteTime             = 0.0;

   std::shared_ptr<WeightsFile> mWeightsFile;
};

} // namespace PV

#endif // WEIGHTSPAIR_HPP_
