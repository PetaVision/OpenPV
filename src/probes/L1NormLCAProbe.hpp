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
   L1NormLCAProbe(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);
   virtual ~L1NormLCAProbe() {}

  protected:
   L1NormLCAProbe() {}

   virtual Response::Status allocateDataStructures() override;

   virtual void createProbeLocal(
        std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults) override;
   virtual void createEnergyProbeComponent(
        std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults) override;
   void initialize(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);
}; // class L1NormLCAProbe

} /* namespace PV */

#endif /* L1NORMLCAPROBE_HPP_ */
