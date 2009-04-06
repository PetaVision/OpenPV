/*
 * StatsProbe.hpp
 *
 *  Created on: Mar 10, 2009
 *      Author: rasmussn
 */

#ifndef STATSPROBE_HPP_
#define STATSPROBE_HPP_

#include "PVLayerProbe.hpp"

namespace PV {

class StatsProbe: public PV::PVLayerProbe {
public:
   StatsProbe(const char * filename, PVBufType type, const char * msg);
   StatsProbe(PVBufType type, const char * msg);
   virtual ~StatsProbe();

   virtual int outputState(float time, PVLayer * l);

protected:
   PV::PVBufType type;
   char * msg;
};

}

#endif /* STATSPROBE_HPP_ */
