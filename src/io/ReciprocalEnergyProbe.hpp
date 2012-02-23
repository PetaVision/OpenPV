/*
 * ReciprocalEnergyProbe.hpp
 *
 *  Created on: Feb 17, 2012
 *      Author: pschultz
 */

#ifndef RECIPROCALENERGYPROBE_HPP_
#define RECIPROCALENERGYPROBE_HPP_

#include "BaseConnectionProbe.hpp"
#include "../connections/ReciprocalConn.hpp"

namespace PV {

class ReciprocalEnergyProbe: public PV::BaseConnectionProbe {
public:
   // public methods
   ReciprocalEnergyProbe(const char * probename, const char * filename, HyPerCol * hc);
   virtual ~ReciprocalEnergyProbe();
   virtual int outputState(float timef, HyPerConn * c);

protected:
   ReciprocalEnergyProbe();
   int initialize(const char * probename, const char * filename, HyPerCol * hc);

private:
   int initialize_base();
};

} /* namespace PV */
#endif /* RECIPROCALENERGYPROBE_HPP_ */
