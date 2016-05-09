/*
 * ReciprocalEnergyProbe.hpp
 *
 *  Created on: Feb 17, 2012
 *      Author: pschultz
 */

#ifndef RECIPROCALENERGYPROBE_HPP_
#define RECIPROCALENERGYPROBE_HPP_

#include "ConnFunctionProbe.hpp"
#include "../connections/ReciprocalConn.hpp"

namespace PV {

class ReciprocalEnergyProbe : public ConnFunctionProbe {
public:
   // public methods
   ReciprocalEnergyProbe(const char * probename, HyPerCol * hc);
   virtual ~ReciprocalEnergyProbe();
   virtual int communicate();
   virtual int allocateProbe();
   virtual double evaluate(double timed);

protected:
   ReciprocalEnergyProbe();
   int initialize(const char * probename, HyPerCol * hc);

private:
   int initialize_base();

private:
   ReciprocalConn * targetRecipConn; // targetConn, dynamic_cast to ReciprocalConn
};

} /* namespace PV */
#endif /* RECIPROCALENERGYPROBE_HPP_ */
