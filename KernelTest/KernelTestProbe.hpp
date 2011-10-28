/*
 * KernelTestProbe.hpp
 *
 *  Created on: Sep 1, 2011
 *      Author: gkenyon
 */

#ifndef KERNELTESTPROBE_HPP_
#define KERNELTESTPROBE_HPP_

#include "../PetaVision/src/io/StatsProbe.hpp"

namespace PV {

class KernelTestProbe: public PV::StatsProbe {
public:
   KernelTestProbe(const char * filename, HyPerCol * hc, const char * msg);
   KernelTestProbe(const char * msg);

   virtual int outputState(float time, HyPerLayer * l	);

};

} /* namespace PV */
#endif /* KERNELTESTPROBE_HPP_ */
