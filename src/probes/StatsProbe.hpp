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

class StatsProbe : public PV::LayerProbe {
  public:
   StatsProbe(const char *name, HyPerCol *hc);
   virtual ~StatsProbe();

   virtual Response::Status outputState(double timef) override;
   virtual int checkpointTimers(PrintStream &timerstream);

  protected:
   StatsProbe();
   int initialize(const char *name, HyPerCol *hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_buffer(enum ParamsIOFlag ioFlag);
   virtual void ioParam_nnzThreshold(enum ParamsIOFlag ioFlag);
   void requireType(PVBufType requiredType);

   /**
    * StatsProbe sets numValues to -1, indicating that the getValues and getValue
    * methods don't
    * work.
    * StatsProbe is an old probe that might be deprecated in favor of more
    * getValues-friendly
    * probes.
    */
   virtual void initNumValues() override;

   virtual Response::Status registerData(Checkpointer *checkpointer) override;

   /**
    * Implements needRecalc() for StatsProbe to always return false (getValues
    * and getValue methods
    * should not be used).
    */
   virtual bool needRecalc(double timevalue) override { return false; }

   /**
    * Implements calcValues() for StatsProbe to always fail (getValues and
    * getValue methods should
    * not be used).
    */
   virtual void calcValues(double timevalue) override {
      Fatal().printf("%s does not use calcValues.\n", getDescription_c());
   }

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
