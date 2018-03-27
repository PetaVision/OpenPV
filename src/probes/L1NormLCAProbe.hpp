/*
 * L1NormLCAProbe.hpp
 *
 *  Created on: Oct 9, 2015
 *      Author: pschultz
 */

#ifndef L1NORMLCAPROBE_HPP_
#define L1NORMLCAPROBE_HPP_

#include "L1NormProbe.hpp"

namespace PV {

/**
 * A special case of L1NormProbe probe, to be used when the target layer is an
 * LCA layer
 * with a soft-threshold transfer function.  The corresponding cost function is
 * the norm
 * measured by L1NormProbe, with coefficient equal to the LCA layer's VThresh.
 */
class L1NormLCAProbe : public L1NormProbe {
  public:
   L1NormLCAProbe(const char *name, HyPerCol *hc);
   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   virtual ~L1NormLCAProbe() {}

  protected:
   L1NormLCAProbe();
   int initialize(const char *name, HyPerCol *hc) { return L1NormProbe::initialize(name, hc); }

   /**
    * L1NormLCAProbe does not read coefficient from its own params group,
    * but takes it from VThresh of the target layer.
    */
   virtual void ioParam_coefficient(enum ParamsIOFlag ioFlag) override {
   } // coefficient is set from targetLayer during communicateInitInfo.

  private:
   int initialize_base() { return PV_SUCCESS; }
}; // end class L1NormLCAProbe

} /* namespace PV */

#endif /* L1NORMLCAPROBE_HPP_ */
