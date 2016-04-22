/*
 * L0NormLCAProbe.hpp
 *
 *  Created on: Oct 9, 2015
 *      Author: pschultz
 */

#ifndef L0NORMLCAPROBE_HPP_
#define L0NORMLCAPROBE_HPP_

#include "L0NormProbe.hpp"

namespace PV {

/**
 * A special case of L0NormProbe probe, to be used when the target layer is an LCA layer
 * with a hard-threshold transfer function.  The corresponding cost function is the norm
 * measured by L0NormProbe, with coefficient Vth^2/2, where Vth is the target LCA layer's VThresh.
 */
class L0NormLCAProbe: public L0NormProbe {
public:
   L0NormLCAProbe(const char * probeName, HyPerCol * hc);
   virtual int communicateInitInfo();
   virtual ~L0NormLCAProbe() {}

protected:
   L0NormLCAProbe();
   int initL0NormLCAProbe(const char * probeName, HyPerCol * hc)  { return initL0NormProbe(probeName, hc); }

   /**
    * L0NormLCAProbe does not read coefficient from its own params group,
    * but computes it in terms of VThresh of the target layer.
    */
   virtual void ioParam_coefficient(enum ParamsIOFlag ioFlag) {} // coefficient is set from targetLayer during communicateInitInfo.

private:
   int initialize_base() { return PV_SUCCESS; }
}; // end class L0NormLCAProbe

BaseObject * createL0NormLCAProbe(char const * name, HyPerCol * hc);

} /* namespace PV */

#endif /* L0NORMLCAPROBE_HPP_ */
