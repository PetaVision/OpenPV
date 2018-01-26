/*
 * KernelTestProbe.hpp
 *
 *  Created on: Sep 1, 2011
 *      Author: gkenyon
 */

#ifndef KERNELTESTPROBE_HPP_
#define KERNELTESTPROBE_HPP_

#include "probes/StatsProbe.hpp"

namespace PV {

class KernelTestProbe : public PV::StatsProbe {
  public:
   KernelTestProbe(const char *name, HyPerCol *hc);

   virtual Response::Status outputState(double timestamp) override;

  protected:
   int initialize(const char *name, HyPerCol *hc);
   void ioParam_buffer(enum ParamsIOFlag ioFlag) override;

  private:
   int initialize_base();
};

} /* namespace PV */
#endif /* KERNELTESTPROBE_HPP_ */
