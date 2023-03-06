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
 * A special case of L1NormProbe, to be used when the target layer is an
 * LCA layer with a hard-threshold transfer function.  The corresponding cost
 * function is the norm measured by L1NormProbe, with coefficient Vth, where
 * Vth is the target LCA layer's VThresh.
 */
class L1NormLCAProbe : public L1NormProbe {
  public:
   L1NormLCAProbe(const char *name, PVParams *params, Communicator const *comm);
   virtual ~L1NormLCAProbe() {}

  protected:
   L1NormLCAProbe() {}
   virtual void createEnergyProbeComponent(char const *name, PVParams *params) override;
   void initialize(const char *name, PVParams *params, Communicator const *comm);
}; // class L1NormLCAProbe

} /* namespace PV */

#endif /* L1NORMLCAPROBE_HPP_ */
