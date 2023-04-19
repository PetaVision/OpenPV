#ifndef VMEMBRANEBUFFERSTATSPROBELOCAL_HPP_
#define VMEMBRANEBUFFERSTATSPROBELOCAL_HPP_

#include "io/PVParams.hpp"
#include "probes/StatsProbeLocal.hpp"

namespace PV {

class VMembraneBufferStatsProbeLocal : public StatsProbeLocal {
  public:
   VMembraneBufferStatsProbeLocal(char const *objName, PVParams *params);
   virtual ~VMembraneBufferStatsProbeLocal(){};

  protected:
   VMembraneBufferStatsProbeLocal() {}
   void initialize(char const *objName, PVParams *params);

}; // class VMembraneBufferStatsProbeLocal

} // namespace PV

#endif // VMEMBRANEBUFFERSTATSPROBELOCAL_HPP_
