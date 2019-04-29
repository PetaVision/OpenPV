/*
 * HyPerConnCheckpointerProbe.hpp
 *
 *  Created on: Jan 5, 2017
 *      Author: pschultz
 */

#ifndef POOLINGCONNCHECKPOINTERTESTPROBE_HPP_
#define POOLINGCONNCHECKPOINTERTESTPROBE_HPP_

#include "probes/ColProbe.hpp"

#include "CorrectState.hpp"
#include "components/BasePublisherComponent.hpp"
#include "connections/PoolingConn.hpp"
#include "layers/HyPerLayer.hpp"
#include "layers/InputLayer.hpp"

class PoolingConnCheckpointerTestProbe : public PV::ColProbe {
  public:
   /**
    * Public constructor for the PoolingConnCheckpointerTestProbe class.
    */
   PoolingConnCheckpointerTestProbe(
         const char *name,
         PV::PVParams *params,
         PV::Communicator const *comm);

   /**
    * Destructor for the PoolingConnCheckpointerTestProbe class.
    */
   virtual ~PoolingConnCheckpointerTestProbe();

   /**
    * Implementation of the calcValues method. This probe does not compute
    * any values that are available to other objects through getValues().
    */
   virtual void calcValues(double timevalue) override {}

   bool getTestFailed() const { return mTestFailed; }

  protected:
   void initialize(const char *name, PV::PVParams *params, PV::Communicator const *comm);
   virtual void ioParam_textOutputFlag(enum PV::ParamsIOFlag ioFlag) override;
   virtual PV::Response::Status
   communicateInitInfo(std::shared_ptr<PV::CommunicateInitInfoMessage const> message) override;
   virtual PV::Response::Status
   initializeState(std::shared_ptr<PV::InitializeStateMessage const> message) override;
   virtual PV::Response::Status readStateFromCheckpoint(PV::Checkpointer *checkpointer) override;
   virtual bool needRecalc(double timevalue) override { return true; }
   virtual double referenceUpdateTime(double simTime) const override { return simTime; }
   virtual PV::Response::Status outputState(double simTime, double deltaTime) override;

  private:
   PoolingConnCheckpointerTestProbe();

   /**
    * Sets the connection data member, and checks that the connection's parameters are
    * consistent with those expected by the test. Returns either PV_SUCCESS or PV_POSTPONE.
    */
   PV::Response::Status initConnection(PV::ObserverTable const *objectTable);

   /**
    * Sets the input layer data member, and checks that the input layer's parameters are
    * consistent with those expected by the test. Returns either PV_SUCCESS or PV_POSTPONE.
    */
   PV::Response::Status initInputPublisher(PV::ObserverTable const *objectTable);

   /**
    * Sets the output layer data member, and checks that the output layer's parameters are
    * consistent with those expected by the test. Returns either PV_SUCCESS or PV_POSTPONE.
    */
   PV::Response::Status initOutputPublisher(PV::ObserverTable const *objectTable);

   /**
    * Checks whether the given object has finished its communicateInitInfo stage, and
    * returns PV_SUCCESS if it has, or PV_POSTPONE if it has not.
    */
   PV::Response::Status checkCommunicatedFlag(PV::BaseObject *dependencyObject);

   int calcUpdateNumber(double timevalue);
   void initializeCorrectValues(double timevalue);

   bool verifyLayer(
         PV::BasePublisherComponent *layer,
         PV::Buffer<float> const &correctValueBuffer,
         double timevalue);

   // Data members
  protected:
   int mStartingUpdateNumber                    = 0;
   bool mValuesSet                              = false;
   PV::BasePublisherComponent *mInputPublisher  = nullptr;
   PV::BasePublisherComponent *mOutputPublisher = nullptr;
   PV::PoolingConn *mConnection                 = nullptr;
   CorrectState *mCorrectState                  = nullptr;
   bool mTestFailed                             = false;
};

#endif // POOLINGCONNCHECKPOINTERTESTPROBE_HPP_
