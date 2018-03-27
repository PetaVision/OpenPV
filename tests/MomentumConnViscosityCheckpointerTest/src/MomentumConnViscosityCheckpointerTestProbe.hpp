/*
 * MomentumConnSimpleCheckpointerProbe.hpp
 *
 *  Created on: Jan 6, 2017
 *      Author: pschultz
 */

#ifndef MOMENTUMCONNVISCOSITYCHECKPOINTERTESTPROBE_HPP_
#define MOMENTUMCONNVISCOSITYCHECKPOINTERTESTPROBE_HPP_

#include "probes/ColProbe.hpp"

#include "CorrectState.hpp"
#include "connections/MomentumConn.hpp"
#include "layers/HyPerLayer.hpp"
#include "layers/InputLayer.hpp"

class MomentumConnViscosityCheckpointerTestProbe : public PV::ColProbe {
  public:
   /**
    * Public constructor for the MomentumConnViscosityCheckpointerTestProbe class.
    */
   MomentumConnViscosityCheckpointerTestProbe(const char *name, PV::HyPerCol *hc);

   /**
    * Destructor for the MomentumConnViscosityCheckpointerTestProbe class.
    */
   virtual ~MomentumConnViscosityCheckpointerTestProbe();

   /**
    * Implementation of the calcValues method. This probe does not compute
    * any values that are available to other objects through getValues().
    */
   virtual void calcValues(double timevalue) override {}

   bool getTestFailed() const { return mTestFailed; }

  protected:
   int initialize(const char *name, PV::HyPerCol *hc);
   virtual void ioParam_textOutputFlag(enum PV::ParamsIOFlag ioFlag) override;
   virtual PV::Response::Status
   communicateInitInfo(std::shared_ptr<PV::CommunicateInitInfoMessage const> message) override;
   virtual PV::Response::Status readStateFromCheckpoint(PV::Checkpointer *checkpointer) override;
   virtual bool needRecalc(double timevalue) override { return true; }
   virtual double referenceUpdateTime() const override { return parent->simulationTime(); }
   virtual PV::Response::Status outputState(double timestamp) override;

  private:
   MomentumConnViscosityCheckpointerTestProbe();
   int initialize_base();

   /**
    * Sets the input layer data member, and checks that the input layer's parameters are
    * consistent with those expected by the test. Returns either SUCCESS or POSTPONE.
    */
   PV::Response::Status
   initInputLayer(std::shared_ptr<PV::CommunicateInitInfoMessage const> message);

   /**
    * Sets the output layer data member, and checks that the output layer's parameters are
    * consistent with those expected by the test. Returns either SUCCESS or POSTPONE.
    */
   PV::Response::Status
   initOutputLayer(std::shared_ptr<PV::CommunicateInitInfoMessage const> message);

   /**
    * Sets the connection data member, and checks that the connection's parameters are
    * consistent with those expected by the test. Returns either SUCCESS or POSTPONE.
    */
   PV::Response::Status
   initConnection(std::shared_ptr<PV::CommunicateInitInfoMessage const> message);

   /**
    * Checks whether the given object has finished its communicateInitInfo stage, and
    * returns SUCCESS if it has, or POSTPONE if it has not.
    */
   PV::Response::Status checkCommunicatedFlag(PV::BaseObject *dependencyObject);

   int calcUpdateNumber(double timevalue);
   void initializeCorrectValues(double timevalue);

   bool verifyConnection(
         PV::MomentumConn *connection,
         CorrectState const *correctState,
         double timevalue);
   bool
   verifyConnValue(double timevalue, float observed, float correct, char const *valueDescription);
   bool verifyLayer(PV::HyPerLayer *layer, float correctValue, double timevalue);

   // Data members
  protected:
   int mStartingUpdateNumber     = 0;
   bool mValuesSet               = false;
   PV::InputLayer *mInputLayer   = nullptr;
   PV::HyPerLayer *mOutputLayer  = nullptr;
   PV::MomentumConn *mConnection = nullptr;
   CorrectState *mCorrectState   = nullptr;
   bool mTestFailed              = false;
};

#endif // MOMENTUMCONNVISCOSITYCHECKPOINTERTESTPROBE_HPP_
