/*
 * MomentumConnSimpleCheckpointerProbe.hpp
 *
 *  Created on: Jan 6, 2017
 *      Author: pschultz
 */

#ifndef MOMENTUMCONNSIMPLECHECKPOINTERTESTPROBE_HPP_
#define MOMENTUMCONNSIMPLECHECKPOINTERTESTPROBE_HPP_

#include "probes/ColProbe.hpp"

#include "CorrectState.hpp"
#include "connections/MomentumConn.hpp"
#include "layers/HyPerLayer.hpp"
#include "layers/InputLayer.hpp"

class MomentumConnSimpleCheckpointerTestProbe : public PV::ColProbe {
  public:
   /**
    * Public constructor for the MomentumConnSimpleCheckpointerTestProbe class.
    */
   MomentumConnSimpleCheckpointerTestProbe(const char *probeName, PV::HyPerCol *hc);

   /**
    * Destructor for the MomentumConnSimpleCheckpointerTestProbe class.
    */
   virtual ~MomentumConnSimpleCheckpointerTestProbe();

   /**
    * Implementation of the calcValues method. This probe does not compute
    * any values that are available to other objects through getValues().
    */
   virtual int calcValues(double timevalue) { return PV_SUCCESS; }

   bool getTestFailed() const { return mTestFailed; }

  protected:
   int initialize(const char *probeName, PV::HyPerCol *hc);
   virtual void ioParam_textOutputFlag(enum PV::ParamsIOFlag ioFlag) override;
   virtual int communicateInitInfo() override;
   virtual int readStateFromCheckpoint(PV::Checkpointer *checkpointer) override;
   virtual bool needRecalc(double timevalue) override { return true; }
   virtual double referenceUpdateTime() const override { return parent->simulationTime(); }
   virtual int outputState(double timevalue);

  private:
   MomentumConnSimpleCheckpointerTestProbe();
   int initialize_base();

   /**
    * Sets the input layer data member, and checks that the input layer's parameters are
    * consistent with those expected by the test. Returns either PV_SUCCESS or PV_POSTPONE.
    */
   int initInputLayer();

   /**
    * Sets the output layer data member, and checks that the output layer's parameters are
    * consistent with those expected by the test. Returns either PV_SUCCESS or PV_POSTPONE.
    */
   int initOutputLayer();

   /**
    * Sets the connection data member, and checks that the connection's parameters are
    * consistent with those expected by the test. Returns either PV_SUCCESS or PV_POSTPONE.
    */
   int initConnection();

   /**
    * Checks whether the given object has finished its communicateInitInfo stage, and
    * returns PV_SUCCESS if it has, or PV_POSTPONE if it has not.
    */
   int checkCommunicatedFlag(PV::BaseObject *dependencyObject);

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
   bool mValuesSet               = 0.0;
   PV::InputLayer *mInputLayer   = nullptr;
   PV::HyPerLayer *mOutputLayer  = nullptr;
   PV::MomentumConn *mConnection = nullptr;
   CorrectState *mCorrectState   = nullptr;
   bool mTestFailed              = false;
};

#endif // MOMENTUMCONNSIMPLECHECKPOINTERTESTPROBE_HPP_
