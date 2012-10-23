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
   CloneKernelConnTestProbe(const char * filename, HyPerLayer * layer, const char * msg);
   CloneKernelConnTestProbe(HyPerLayer * layer, const char * msg);

   virtual int outputState(double timed);

};

} /* namespace PV */
#endif /* CLONEKERNELCONNTESTPROBE_HPP_ */
