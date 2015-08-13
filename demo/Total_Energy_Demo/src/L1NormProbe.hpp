/*
 * L1NormProbe.hpp
 *
 *  Created on: Aug 11, 2015
 *      Author: pschultz
 */

#ifndef L1NORMPROBE_HPP_
#define L1NORMPROBE_HPP_

#include <io/LayerProbe.hpp>

namespace PV {
class L1NormProbe : public LayerProbe {
public:
   L1NormProbe(const char * probeName, HyPerCol * hc);
   virtual ~L1NormProbe();
   virtual int getValues(double timevalue, std::vector<double> * values);
   virtual double getValue(double timevalue, int index);

protected:
   L1NormProbe();
   int initL1NormProbe(const char * probeName, HyPerCol * hc);
   virtual double getValueInternal(double timevalue, int index);
   virtual int outputState(double timevalue);

private:
   int initL1NormProbe_base() {return PV_SUCCESS;}
}; // end class L1NormProbe

}  // end namespace PV

#endif /* L1NORMPROBE_HPP_ */
