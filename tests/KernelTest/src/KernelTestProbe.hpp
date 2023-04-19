/*
 * KernelTestProbe.hpp
 *
 *  Created on: Sep 1, 2011
 *      Author: gkenyon
 */

#ifndef KERNELTESTPROBE_HPP_
#define KERNELTESTPROBE_HPP_

#include "columns/Communicator.hpp"
#include "io/PVParams.hpp"
#include "probes/StatsProbeImmediate.hpp"

namespace PV {

class KernelTestProbe : public PV::StatsProbeImmediate {
  public:
   KernelTestProbe(const char *name, PVParams *params, Communicator const *comm);

  protected:
   virtual void checkStats() override;
   virtual void createProbeLocal(char const *name, PVParams *params) override;
   void initialize(const char *name, PVParams *params, Communicator const *comm);
};

} /* namespace PV */
#endif /* KERNELTESTPROBE_HPP_ */
