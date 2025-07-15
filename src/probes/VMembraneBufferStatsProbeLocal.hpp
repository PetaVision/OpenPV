#ifndef VMEMBRANEBUFFERSTATSPROBELOCAL_HPP_
#define VMEMBRANEBUFFERSTATSPROBELOCAL_HPP_

#include "probes/StatsProbeLocal.hpp"

namespace PV {

class VMembraneBufferStatsProbeLocal : public StatsProbeLocal {
  public:
   VMembraneBufferStatsProbeLocal(std::shared_ptr<ParamsIO> paramsIO);
   virtual ~VMembraneBufferStatsProbeLocal() {};

  protected:
   VMembraneBufferStatsProbeLocal() {}
   void initialize(std::shared_ptr<ParamsIO> paramsIO);

}; // class VMembraneBufferStatsProbeLocal

} // namespace PV

#endif // VMEMBRANEBUFFERSTATSPROBELOCAL_HPP_
