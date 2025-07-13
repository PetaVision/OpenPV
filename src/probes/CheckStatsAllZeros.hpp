#ifndef CHECKSTATSALLZEROS_HPP_
#define CHECKSTATSALLZEROS_HPP_

#include "params/ParamsIO.hpp"
#include "probes/ProbeData.hpp"
#include "probes/StatsProbeTypes.hpp"

#include <map>
#include <string>

namespace PV {

class CheckStatsAllZeros {
  protected:
   void ioParam_exitOnFailure(ParamsIOSwitch ioSwitch, std::shared_ptr<ParamsIO> paramsIO);
   void ioParam_immediateExitOnFailure(ParamsIOSwitch ioSwitch, std::shared_ptr<ParamsIO> paramsIO);

  public:
   CheckStatsAllZeros(std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults);
   virtual ~CheckStatsAllZeros();

   virtual std::map<int, LayerStats const> checkStats(ProbeData<LayerStats> const &batchProbeData);
   void cleanup();
   bool foundNonzero() const { return !mFirstFailure.empty(); }
   void ioParamsFillGroup(ParamsIOSwitch ioSwitch, std::shared_ptr<ParamsIO> paramsIO);

   bool getExitOnFailure() const { return mExitOnFailure; }
   double getFirstFailureTime() const { return mFirstFailureTime; }
   bool getImmediateExitOnFailure() const { return mImmediateExitOnFailure; }

   std::string const &getName() const { return mParams->getName(); }
   std::string const &getKeyword() const { return mParams->getKeyword(); }
   char const *getName_c() const { return mParams->getName().c_str(); }
   char const *getKeyword_c() const { return mParams->getKeyword().c_str(); }

  protected:
   void setFirstFailure(std::map<int, LayerStats const> const &failureMap, double failureTime);
   std::string errorMessage(
         std::map<int, LayerStats const> const &badCounts,
         double badTime,
         std::string const &baseMessage) const;

  protected:
   std::shared_ptr<ParamGroup> mParams;
   std::shared_ptr<ParamGroup> mDefaults;

  private:
   bool mExitOnFailure = true;
   std::map<int, LayerStats const> mFirstFailure;
   double mFirstFailureTime;
   bool mImmediateExitOnFailure = true;
}; // class CheckStatsAllZeros

} // namespace PV

#endif // CHECKSTATSALLZEROS_HPP_
