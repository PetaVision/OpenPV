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
   double sum;
   float fMin, fMax;
   float avg;

private:
   int initMessage(const char * msg);
};

}

#endif /* STATSPROBE_HPP_ */
