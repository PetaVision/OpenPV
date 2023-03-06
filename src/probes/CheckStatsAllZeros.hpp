#ifndef CHECKSTATSALLZEROS_HPP_
#define CHECKSTATSALLZEROS_HPP_

#include "io/PVParams.hpp"
#include "probes/ProbeData.hpp"
#include "probes/StatsProbeTypes.hpp"

#include <map>
#include <string>

namespace PV {

class CheckStatsAllZeros {
  protected:
   void ioParam_exitOnFailure(enum ParamsIOFlag ioFlag);
   void ioParam_immediateExitOnFailure(enum ParamsIOFlag ioFlag);

  public:
   CheckStatsAllZeros(char const *objName, PVParams *params);
   virtual ~CheckStatsAllZeros();

   virtual std::map<int, LayerStats const> checkStats(ProbeData<LayerStats> const &batchProbeData);
   void cleanup();
   bool foundNonzero() const { return !mFirstFailure.empty(); }
   void ioParamsFillGroup(enum ParamsIOFlag ioFlag);

   bool getExitOnFailure() const { return mExitOnFailure; }
   double getFirstFailureTime() const { return mFirstFailureTime; }
   bool getImmediateExitOnFailure() const { return mImmediateExitOnFailure; }
   std::string const &getName() const { return mName; }

  protected:
   void setFirstFailure(std::map<int, LayerStats const> const &failureMap, double failureTime);
   std::string errorMessage(
         std::map<int, LayerStats const> const &badCounts,
         double badTime,
         std::string const &baseMessage) const;

  private:
   bool mExitOnFailure = true;
   std::map<int, LayerStats const> mFirstFailure;
   double mFirstFailureTime;
   bool mImmediateExitOnFailure = true;
   std::string mName;
   PVParams *mParams;
}; // class CheckStatsAllZeros

} // namespace PV

#endif // CHECKSTATSALLZEROS_HPP_
