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
   StatsProbe(const char * filename, HyPerLayer * layer, const char * msg);
   StatsProbe(HyPerLayer * layer, const char * msg);
   StatsProbe(const char * filename, HyPerLayer * layer, PVBufType type, const char * msg);
   StatsProbe(HyPerLayer * layer, PVBufType type, const char * msg);
   virtual ~StatsProbe();

   virtual int outputState(double timef);

protected:
   StatsProbe();
   int initStatsProbe(const char * filename, HyPerLayer * layer, PVBufType type, const char * msg);
   PV::PVBufType type;
   char * msg;
   double sum, sum2;
   int nnz;
   float fMin, fMax;
   float avg, sigma;
   Timer * iotimer;   // A timer for the i/o part of outputState
   Timer * mpitimer;  // A timer for the MPI part of outputState
   Timer * comptimer; // A timer for the basic computation of outputState

private:
   int initStatsProbe_base();
   int initMessage(const char * msg);
};

}

#endif /* STATSPROBE_HPP_ */
