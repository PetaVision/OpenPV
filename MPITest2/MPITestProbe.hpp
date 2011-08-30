/*
 * customStatsProbe.hpp
 *
 *  Created on: Mar 10, 2009
 *      Author: garkenyon
 */

#ifndef MPITESTPROBE_HPP_
#define MPITESTPROBE_HPP_

#include "../PetaVision/src/io/StatsProbe.hpp"

namespace PV {

class MPITestProbe: public PV::StatsProbe {
public:
   MPITestProbe(const char * filename, HyPerCol * hc, PVBufType type, const char * msg);
   MPITestProbe(PVBufType type, const char * msg);

   virtual int outputState(float time, HyPerLayer * l);

protected:
   double cumSum;
   double cumAvg;
};

}

#endif /* MPITESTPROBE_HPP_ */
