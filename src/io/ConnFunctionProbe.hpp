/*
 * ConnFunctionProbe.hpp
 *
 *  Created on: Mar 23, 2012
 *      Author: pschultz
 */

#ifndef CONNFUNCTIONPROBE_HPP_
#define CONNFUNCTIONPROBE_HPP_

#include "BaseConnectionProbe.hpp"
#include "../connections/HyPerConn.hpp"

namespace PV {

class ConnFunctionProbe: public BaseConnectionProbe {
public:
   ConnFunctionProbe(const char * probename, const char * filename, HyPerConn * conn);
   virtual ~ConnFunctionProbe();
   virtual int outputState(float timef);
   virtual double evaluate(float timef) {return 0.0f;}

protected:
   ConnFunctionProbe();
   int initialize(const char * probename, const char * filename, HyPerConn * conn);

private:
   int initialize_base();
};

}  // end namespace PV

#endif /* CONNFUNCTIONPROBE_HPP_ */
