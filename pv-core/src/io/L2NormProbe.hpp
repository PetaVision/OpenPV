/*
 * L2NormProbe.hpp
 *
 *  Created on: Aug 11, 2015
 *      Author: pschultz
 */

#ifndef L2NORMPROBE_HPP_
#define L2NORMPROBE_HPP_

#include "AbstractNormProbe.hpp"

namespace PV {
class L2NormProbe : public AbstractNormProbe {
public:
   L2NormProbe(const char * probeName, HyPerCol * hc);
   virtual ~L2NormProbe();
   virtual int getValues(double timevalue, std::vector<double> * values);
   virtual double getValue(double timevalue, int index);

protected:
   L2NormProbe();
   int initL2NormProbe(const char * probeName, HyPerCol * hc);
   virtual double getValueInternal(double timevalue, int index);
   virtual int outputState(double timevalue);
   
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_exponent(enum ParamsIOFlag ioFlag);

private:
   int initL2NormProbe_base() {return PV_SUCCESS;}

// Member variables
   double exponent;
}; // end class L2NormProbe

}  // end namespace PV

#endif /* L2NORMPROBE_HPP_ */
