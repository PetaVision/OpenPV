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
   virtual void ioParam_isViscosity(enum ParamsIOFlag ioFlag);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);

protected:
   int initialize(const char * probename, HyPerCol * hc);
   int isViscosity;
}; // end class MomentumConnTestProbe

BaseObject * createMomentumConnTestProbe(char const * name, HyPerCol * hc);

}  // end namespace PV

#endif /* PLASTICCONNTESTPROBE_HPP_ */
