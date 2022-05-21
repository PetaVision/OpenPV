/*
 * StatsProbe.hpp
 *
 *  Created on: Mar 10, 2009
 *      Author: rasmussn
 */

#ifndef STATSPROBE_HPP_
#define STATSPROBE_HPP_

#include "LayerProbe.hpp"

namespace PV {

class StatsProbe : public LayerProbe {
  public:
   StatsProbe(const char *name, PVParams *params, Communicator const *comm);
   virtual ~StatsProbe();

   virtual int checkpointTimers(PrintStream &timerstream);

  protected:
   StatsProbe();
   void initialize(const char *name, PVParams *params, Communicator const *comm);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief statsFlag: StatsProbe does not use statsFlag.
    */
   virtual void ioParam_statsFlag(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_buffer(enum ParamsIOFlag ioFlag);
   virtual void ioParam_nnzThreshold(enum ParamsIOFlag ioFlag);
   void requireType(PVBufType requiredType);

   /**
    * StatsProbe sets numValues to -1, indicating that the getValues methods don't work.
    * StatsProbe is an old probe that might eventually be deprecated in favor of more
    * getValues-friendly probes.
    */
   virtual void initNumValues() override;

   virtual Response::Status allocateDataStructures() override;

   virtual Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;

   /**
    * Implements needRecalc() for StatsProbe to always return false
    * (the getValues() function member should not be used).
    */
   virtual bool needRecalc(double timevalue) override { return false; }

   /**
    * Implements calcValues() for StatsProbe to always fail
    * (the getValues() function member should not be used).
    */
   virtual void calcValues(double timevalue) override {
      Fatal().printf("%s does not use calcValues.\n", getDescription_c());
   }

   float const *retrieveActivityBuffer();
   float const *retrieveVBuffer();

   virtual Response::Status outputState(double simTime, double deltaTime) override;
   virtual Response::Status outputStateStats(double simTime, double deltaTime) override;

   // Member variables
   PVBufType type;
   double *sum;
   double *sum2;
   int *nnz;
   float *fMin;
   float *fMax;
   float *avg;
   float *sigma;

   float nnzThreshold;
   Timer *iotimer; // A timer for the i/o part of outputState
   Timer *mpitimer; // A timer for the MPI part of outputState
   Timer *comptimer; // A timer for the basic computation of outputState

  private:
   int initialize_base();
   void resetStats();
}; // end class StatsProbe
}

#endif /* STATSPROBE_HPP_ */
