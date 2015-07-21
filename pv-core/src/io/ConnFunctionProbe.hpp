/*
 * ConnFunctionProbe.hpp
 *
 *  Created on: Mar 23, 2012
 *      Author: pschultz
 */

#ifndef CONNFUNCTIONPROBE_HPP_
#define CONNFUNCTIONPROBE_HPP_

#include "BaseConnectionProbe.hpp"
#include "GenColProbe.hpp"
#include "../connections/HyPerConn.hpp"

namespace PV {

class ConnFunctionProbe: public BaseConnectionProbe {
public:
   ConnFunctionProbe(const char * probename, HyPerCol * hc);
   virtual ~ConnFunctionProbe();
   virtual int communicate();
   virtual int allocateProbe();
   virtual int outputState(double timef);
   virtual double evaluate(double timef) {return 0.0f;}

protected:
   ConnFunctionProbe();
   int initialize(const char * probename, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_parentGenColProbe(enum ParamsIOFlag ioFlag);

private:
   int initialize_base();

// Member variables
private:
   char * parentGenColName;
   GenColProbe * parentGenCol;
};

}  // end namespace PV

#endif /* CONNFUNCTIONPROBE_HPP_ */
