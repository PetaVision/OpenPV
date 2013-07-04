/*
 * OjaKernelSpikeRateProbe.hpp
 *
 *  Created on: Nov 5, 2012
 *      Author: pschultz
 */

#ifndef OJAKERNELSPIKERATEPROBE_HPP_
#define OJAKERNELSPIKERATEPROBE_HPP_

#include "BaseConnectionProbe.hpp"
#include "../connections/OjaKernelConn.hpp"

namespace PV {

class OjaKernelSpikeRateProbe: public PV::BaseConnectionProbe {
public:
   OjaKernelSpikeRateProbe(const char * probename, const char * filename, HyPerConn * conn);
   virtual ~OjaKernelSpikeRateProbe();
   virtual int outputState(double timed);

protected:
   OjaKernelSpikeRateProbe();
   int initialize_base();

private:
   int initialize(const char * probename, const char * filename, HyPerConn * conn);

protected:
   OjaKernelConn * targetOjaKernelConn;
   const pvdata_t * spikeRate; // Address of the spike rate being probed (not the start of the buffer, the location of the specific rate)
   int xg, yg, feature; // global coordinates of the connection being probed
   bool isInputRate; // true if getting an input rate, false if getting an output rate
   int arbor; // If isInputRate, need to specify arbor
};

} /* namespace PV */
#endif /* OJAKERNELSPIKERATEPROBE_HPP_ */
