/*
 * HyPerConnCheckpointerProbe.hpp
 *
 *  Created on: Jan 5, 2017
 *      Author: pschultz
 */

#ifndef HYPERCONNCHECKPOINTERTESTPROBE_HPP_
#define HYPERCONNCHECKPOINTERTESTPROBE_HPP_

#include "probes/ColProbe.hpp"

#include "connections/HyPerConn.hpp"
#include "layers/HyPerLayer.hpp"
#include "layers/InputLayer.hpp"

class HyPerConnCheckpointerTestProbe : public PV::ColProbe {
  public:
   /**
    * Public constructor for the HyPerConnCheckpointerTestProbe class.
    */
   HyPerConnCheckpointerTestProbe(const char *probeName, PV::HyPerCol *hc);

   /**
    * Destructor for the HyPerConnCheckpointerTestProbe class.
    */
   virtual ~HyPerConnCheckpointerTestProbe();

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
   HyPerConnCheckpointerTestProbe();
   int initialize_base();

   int calcUpdateNumber(double timevalue);
   void nextValues(int j);
   void initializeCorrectValues(double timevalue);

   bool verifyLayer(PV::HyPerLayer *layer, float correctValue, double timevalue);
   bool verifyConnection(PV::HyPerConn *connection, float correctValue, double timevalue);

   // Data members
  protected:
   double mEffectiveStartTime     = 0.0;
   bool mValuesSet                = 0.0;
   int mUpdateNumber              = 0;
   PV::InputLayer *mInputLayer    = nullptr;
   PV::HyPerLayer *mOutputLayer   = nullptr;
   PV::HyPerConn *mConnection     = nullptr;
   float mCorrectWeightValue      = 0.0f;
   float mCorrectInputLayerValue  = 0.0f;
   float mCorrectOutputLayerValue = 0.0f;
   bool mTestFailed               = false;
};

#endif // HYPERCONNCHECKPOINTERTESTPROBE_HPP_
