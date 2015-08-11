/*
 * L1NormProbe.hpp
 *
 *  Created on: Aug 11, 2015
 *      Author: pschultz
 */

#ifndef L1NORMPROBE_HPP_
#define L1NORMPROBE_HPP_

#include <io/LayerFunctionProbe.hpp>
#include "L1NormFunction.hpp"

namespace PV {
class L1NormProbe : public LayerFunctionProbe {
public:
   L1NormProbe(const char * probeName, HyPerCol * hc);
   virtual ~L1NormProbe();
   pvdata_t evaluate();

protected:
   L1NormProbe();
   int initL1NormProbe(const char * probeName, HyPerCol * hc);
   virtual void initFunction();
   virtual int writeState(double timed, HyPerLayer * l, int batchIdx, pvdata_t value);

private:
   int initL1NormProbe_base() {return PV_SUCCESS;}
}; // end class L1NormProbe

}  // end namespace PV

#endif /* L1NORMPROBE_HPP_ */
