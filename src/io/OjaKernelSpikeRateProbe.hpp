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
   OjaKernelSpikeRateProbe(const char * probename, HyPerCol * hc);
   virtual ~OjaKernelSpikeRateProbe();
   virtual int allocateDataStructures();
   virtual int outputState(double timed);

protected:
   OjaKernelSpikeRateProbe();
   int initialize(const char * probename, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_x(enum ParamsIOFlag ioFlag);
   virtual void ioParam_y(enum ParamsIOFlag ioFlag);
   virtual void ioParam_f(enum ParamsIOFlag ioFlag);
   virtual void ioParam_isInputRate(enum ParamsIOFlag ioFlag);
   virtual void ioParam_arbor(enum ParamsIOFlag ioFlag);

private:
   int initialize_base();

protected:
   OjaKernelConn * targetOjaKernelConn;
   const pvdata_t * spikeRate; // Address of the spike rate being probed (not the start of the buffer, the location of the specific rate)
   int xg, yg, feature; // global coordinates of the connection being probed
   bool isInputRate; // true if getting an input rate, false if getting an output rate
   int arbor; // If isInputRate, need to specify arbor
};

} /* namespace PV */
#endif /* OJAKERNELSPIKERATEPROBE_HPP_ */
