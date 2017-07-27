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
 * A special case of L0NormProbe probe, to be used when the target layer is an
 * LCA layer
 * with a hard-threshold transfer function.  The corresponding cost function is
 * the norm
 * measured by L0NormProbe, with coefficient Vth^2/2, where Vth is the target
 * LCA layer's VThresh.
 */
class L0NormLCAProbe : public L0NormProbe {
  public:
   L0NormLCAProbe(const char *name, HyPerCol *hc);
   virtual int
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   virtual ~L0NormLCAProbe() {}

  protected:
   L0NormLCAProbe();
   int initialize(const char *name, HyPerCol *hc) { return L0NormProbe::initialize(name, hc); }

   /**
    * L0NormLCAProbe does not read coefficient from its own params group,
    * but computes it in terms of VThresh of the target layer.
    */
   virtual void ioParam_coefficient(enum ParamsIOFlag ioFlag) override {
   } // coefficient is set from targetLayer during communicateInitInfo.

  private:
   int initialize_base() { return PV_SUCCESS; }
}; // end class L0NormLCAProbe

} /* namespace PV */

#endif /* L0NORMLCAPROBE_HPP_ */
