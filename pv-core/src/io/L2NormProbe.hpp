/*
 * L2NormProbe.hpp
 *
 *  Created on: Nov 19, 2010
 *      Author: pschultz
 */

#ifndef L2NORMPROBE_HPP_
#define L2NORMPROBE_HPP_

#include "LayerFunctionProbe.hpp"
#include "L2NormFunction.hpp"

namespace PV {
class L2NormProbe : public LayerFunctionProbe {
public:
   L2NormProbe(const char * probeName, HyPerCol * hc);
   virtual ~L2NormProbe();
   pvdata_t evaluate();

protected:
   L2NormProbe();
   int initL2NormProbe(const char * probeName, HyPerCol * hc);
   virtual void initFunction();
   virtual int writeState(double timed, HyPerLayer * l, pvdata_t value);

private:
   int initL2NormProbe_base() {return PV_SUCCESS;}
}; // end class L2NormProbe

}  // end namespace PV

#endif /* L2NORMPROBE_HPP_ */
