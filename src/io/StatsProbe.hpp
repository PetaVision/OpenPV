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

protected:
   StatsProbe();
   int initStatsProbe(const char * probeName, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_buffer(enum ParamsIOFlag ioFlag);
   void requireType(PVBufType requiredType);
   PVBufType type;
   double sum, sum2;
   int nnz;
   float fMin, fMax;
   float avg, sigma;
   Timer * iotimer;   // A timer for the i/o part of outputState
   Timer * mpitimer;  // A timer for the MPI part of outputState
   Timer * comptimer; // A timer for the basic computation of outputState

private:
   int initStatsProbe_base();
};

}

#endif /* STATSPROBE_HPP_ */
