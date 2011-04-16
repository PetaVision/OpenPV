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
   StatsProbe(const char * filename, HyPerCol * hc, PVBufType type, const char * msg);
   StatsProbe(PVBufType type, const char * msg);
   virtual ~StatsProbe();

   virtual int outputState(float time, HyPerLayer * l);

protected:
   PV::PVBufType type;
   char * msg;
};

}

#endif /* STATSPROBE_HPP_ */
