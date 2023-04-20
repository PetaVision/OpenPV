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
 * A special case of L0NormProbe, to be used when the target layer is an LCA layer
 * with a hard-threshold transfer function.  The corresponding cost function is
 * the norm measured by L0NormProbe, with coefficient Vth^2/2, where Vth is the
 * target LCA layer's VThresh.
 */
class L0NormLCAProbe : public L0NormProbe {
  public:
   L0NormLCAProbe(const char *name, PVParams *params, Communicator const *comm);
   virtual ~L0NormLCAProbe() {}

  protected:
   L0NormLCAProbe() {}

   virtual Response::Status allocateDataStructures() override;

   virtual void createProbeLocal(char const *name, PVParams *params) override;
   virtual void createEnergyProbeComponent(char const *name, PVParams *params) override;
   void initialize(const char *name, PVParams *params, Communicator const *comm);
}; // class L0NormLCAProbe

} /* namespace PV */

#endif /* L0NORMLCAPROBE_HPP_ */
