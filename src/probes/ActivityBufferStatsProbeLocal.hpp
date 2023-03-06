#ifndef ACTIVITYBUFFERSTATSPROBELOCAL_HPP_
#define ACTIVITYBUFFERSTATSPROBELOCAL_HPP_

#include "io/PVParams.hpp"
#include "probes/StatsProbeLocal.hpp"

namespace PV {

class ActivityBufferStatsProbeLocal : public StatsProbeLocal {
  public:
   ActivityBufferStatsProbeLocal(char const *objName, PVParams *params);
   virtual ~ActivityBufferStatsProbeLocal() {}

  protected:
   ActivityBufferStatsProbeLocal() {}
   void initialize(char const *objName, PVParams *params);

}; // class ActivityBufferStatsProbeLocal

} // namespace PV

#endif // ACTIVITYBUFFERSTATSPROBELOCAL_HPP_
