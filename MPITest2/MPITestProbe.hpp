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
   MPITestProbe(const char * filename, HyPerLayer * layer, const char * msg);
   MPITestProbe(HyPerLayer * layer, const char * msg);

   virtual int outputState(double timed);

protected:
   double cumAvg;
};

}

#endif /* MPITESTPROBE_HPP_ */
