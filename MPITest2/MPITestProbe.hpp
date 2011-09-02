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
   MPITestProbe(const char * filename, HyPerCol * hc, const char * msg);
   MPITestProbe(const char * msg);

   virtual int outputState(float time, HyPerLayer * l);

protected:
   double cumAvg;
};

}

#endif /* MPITESTPROBE_HPP_ */
