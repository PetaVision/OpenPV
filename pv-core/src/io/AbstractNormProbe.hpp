/*
 * AbstractNormProbe.hpp
 *
 *  Created on: Aug 11, 2015
 *      Author: pschultz
 */

#ifndef ABSTRACTNORMPROBE_HPP_
#define ABSTRACTNORMPROBE_HPP_

#include "LayerProbe.hpp"

namespace PV {
class AbstractNormProbe : public LayerProbe {
public:
   AbstractNormProbe(const char * probeName, HyPerCol * hc);
   virtual ~AbstractNormProbe();
   virtual int getValues(double timevalue, std::vector<double> * values);
   virtual double getValue(double timevalue, int index);

protected:
   AbstractNormProbe();
   int initAbstractNormProbe(const char * probeName, HyPerCol * hc);
   virtual double getValueInternal(double timevalue, int index) = 0;
   virtual int outputState(double timevalue);

private:
   int initAbstractNormProbe_base() {return PV_SUCCESS;}
}; // end class AbstractNormProbe

}  // end namespace PV

#endif /* ABSTRACTNORMPROBE_HPP_ */
