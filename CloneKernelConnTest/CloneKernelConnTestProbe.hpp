/*
 * CloneKernelConnTestProbe.hpp
 *
 *  Created on: Feb 24, 2012
 *      Author: peteschultz
 */

#ifndef CLONEKERNELCONNTESTPROBE_HPP_
#define CLONEKERNELCONNTESTPROBE_HPP_

#include "../PetaVision/src/io/StatsProbe.hpp"

namespace PV {

class CloneKernelConnTestProbe: public PV::StatsProbe {
public:
   CloneKernelConnTestProbe(const char * filename, HyPerCol * hc, const char * msg);
   CloneKernelConnTestProbe(const char * msg);

   virtual int outputState(float time, HyPerLayer * l);

};

} /* namespace PV */
#endif /* CLONEKERNELCONNTESTPROBE_HPP_ */
