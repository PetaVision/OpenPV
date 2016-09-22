/*
 * CloneHyPerConnTestProbe.hpp
 *
 *  Created on: Feb 24, 2012
 *      Author: peteschultz
 */

#ifndef CLONEKERNELCONNTESTPROBE_HPP_
#define CLONEKERNELCONNTESTPROBE_HPP_

#include "probes/StatsProbe.hpp"

namespace PV {

class CloneHyPerConnTestProbe: public PV::StatsProbe {
public:
   CloneHyPerConnTestProbe(const char * probeName, HyPerCol * hc);

   virtual int outputState(double timed);

protected:
   int initCloneHyPerConnTestProbe(const char * probeName, HyPerCol * hc);

private:
   int initCloneHyPerConnTestProbe_base();
};


} /* namespace PV */
#endif /* CLONEKERNELCONNTESTPROBE_HPP_ */
