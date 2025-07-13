#ifndef ACTIVITYBUFFERSTATSPROBELOCAL_HPP_
#define ACTIVITYBUFFERSTATSPROBELOCAL_HPP_

#include "probes/StatsProbeLocal.hpp"

namespace PV {

class ActivityBufferStatsProbeLocal : public StatsProbeLocal {
  public:
   ActivityBufferStatsProbeLocal(
         std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults);
   virtual ~ActivityBufferStatsProbeLocal() {}

  protected:
   ActivityBufferStatsProbeLocal() {}
   void initialize(std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults);

}; // class ActivityBufferStatsProbeLocal

} // namespace PV

#endif // ACTIVITYBUFFERSTATSPROBELOCAL_HPP_
