/*
 * MomentumConnTestProbe.hpp
 *
 *  Created on: Mar 10, 2009
 *      Author: garkenyon
 */

#ifndef MOMENTUMCONNTESTPROBE_HPP_
#define MOMENTUMCONNTESTPROBE_HPP_

#include <io/KernelProbe.hpp>

namespace PV {

class MomentumConnTestProbe: public KernelProbe {
public:
   MomentumConnTestProbe(const char * probename, HyPerCol * hc);
   virtual int outputState(double timed);

protected:
   int initialize(const char * probename, HyPerCol * hc);
};

}

#endif /* PLASTICCONNTESTPROBE_HPP_ */
