/*
 * PlasticConnTestProbe.hpp
 *
 *  Created on: Mar 10, 2009
 *      Author: garkenyon
 */

#ifndef PLASTICCONNTESTPROBE_HPP_
#define PLASTICCONNTESTPROBE_HPP_

#include "../PetaVision/src/io/KernelProbe.hpp"

namespace PV {

class PlasticConnTestProbe: public KernelProbe {
public:
   PlasticConnTestProbe(const char * probename, const char * filename, HyPerCol * hc, int kernelIndex, int arborId);

   virtual int outputState(float time, HyPerConn * c);

   virtual ~PlasticConnTestProbe();

protected:
   int initialize(const char * probename, const char * filename, HyPerCol * hc, int kernelIndex, int arborId);

protected:
   bool errorPresent;
};

}

#endif /* PLASTICCONNTESTPROBE_HPP_ */
