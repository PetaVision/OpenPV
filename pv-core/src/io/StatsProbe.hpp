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

class StatsProbe: public PV::LayerProbe {
public:
   StatsProbe(const char * probeName, HyPerCol * hc);
   virtual ~StatsProbe();

   virtual int outputState(double timef);
virtual int checkpointTimers(PV_Stream * timerstream);
protected:
   StatsProbe();
   int initStatsProbe(const char * probeName, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_buffer(enum ParamsIOFlag ioFlag);
   virtual void ioParam_nnzThreshold(enum ParamsIOFlag ioFlag);
   void requireType(PVBufType requiredType);
   
   /**
    * StatsProbe sets numValues to -1, indicating that the getValues and getValue methods don't work.
    * StatsProbe is an old probe that might be deprecated in favor of more getValues-friendly probes.
    */
   virtual int initNumValues();
   
   /**
    * Implements needRecalc() for StatsProbe to always return false (getValues and getValue methods should not be used).
    */
   virtual bool needRecalc(double timevalue) { return false; }
   
   /**
    * Implements calcValues() for StatsProbe to always fail (getValues and getValue methods should not be used).
    */
   virtual int calcValues(double timevalue) { return PV_FAILURE; }

// Member variables
   PVBufType type;
   double* sum;
   double* sum2;
   int* nnz;
   float* fMin;
   float* fMax;
   float* avg;
   float* sigma;

   pvdata_t nnzThreshold;
   Timer * iotimer;   // A timer for the i/o part of outputState
   Timer * mpitimer;  // A timer for the MPI part of outputState
   Timer * comptimer; // A timer for the basic computation of outputState

private:
   int initStatsProbe_base();
   void resetStats();
}; // end class StatsProbe

BaseObject * createStatsProbe(char const * name, HyPerCol * hc);

}

#endif /* STATSPROBE_HPP_ */
