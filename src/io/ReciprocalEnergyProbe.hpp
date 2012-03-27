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
   ReciprocalEnergyProbe(const char * probename, const char * filename, HyPerConn * conn);
   virtual ~ReciprocalEnergyProbe();
   virtual double evaluate(float timef);

protected:
   ReciprocalEnergyProbe();
   int initialize(const char * probename, const char * filename, HyPerConn * conn);

private:
   int initialize_base();

private:
   ReciprocalConn * targetRecipConn;
};

} /* namespace PV */
#endif /* RECIPROCALENERGYPROBE_HPP_ */
