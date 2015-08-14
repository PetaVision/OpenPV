/*
 * L0NormProbe.hpp
 *
 *  Created on: Aug 11, 2015
 *      Author: pschultz
 */

#ifndef L0NORMPROBE_HPP_
#define L0NORMPROBE_HPP_

#include <io/AbstractNormProbe.hpp>

namespace PV {
class L0NormProbe : public AbstractNormProbe {
public:
   L0NormProbe(const char * probeName, HyPerCol * hc);
   virtual ~L0NormProbe();
   virtual int getValues(double timevalue, std::vector<double> * values);
   virtual double getValue(double timevalue, int index);

protected:
   L0NormProbe();
   int initL0NormProbe(const char * probeName, HyPerCol * hc);
   virtual double getValueInternal(double timevalue, int index);
   virtual int outputState(double timevalue);
   
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_nnzThreshold(enum ParamsIOFlag ioFlag);   

private:
   int initL0NormProbe_base() {return PV_SUCCESS;}

protected:
   pvadata_t nnzThreshold;
}; // end class L0NormProbe

}  // end namespace PV

#endif /* L0NORMPROBE_HPP_ */
