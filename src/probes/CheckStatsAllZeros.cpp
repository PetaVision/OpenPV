#include "CheckStatsAllZeros.hpp"

#include "utils/PVAssert.hpp"
#include "utils/PVLog.hpp"

#include <sstream>
#include <utility>

namespace PV {

CheckStatsAllZeros::CheckStatsAllZeros(std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults)
      : mParams(params), mDefaults(defaults) {}

CheckStatsAllZeros::~CheckStatsAllZeros() {}

std::map<int, LayerStats const>
CheckStatsAllZeros::checkStats(ProbeData<LayerStats> const &batchProbeData) {
   int nbatch = static_cast<int>(batchProbeData.size());
   std::map<int, LayerStats const> result;
   for (int b = 0; b < nbatch; ++b) {
      auto &stats = batchProbeData.getValue(b);
      if (stats.mNumNonzero != 0) {
         result.emplace_hint(result.end(), b, stats);
      }
   }
   if (!result.empty()) {
      if (!foundNonzero()) {
         setFirstFailure(result, batchProbeData.getTimestamp());
      }
      auto message = errorMessage(result, batchProbeData.getTimestamp(), "nonzero activity");
      if (mImmediateExitOnFailure) {
         Fatal() << message;
      }
      else {
         ErrorLog() << message;
      }
   }
   return result;
}

void CheckStatsAllZeros::cleanup() {
   if (foundNonzero()) {
      pvAssert(!mImmediateExitOnFailure);
      auto message = errorMessage(mFirstFailure, mFirstFailureTime, "nonzero activity beginning");
      if (mExitOnFailure) {
         Fatal() << message;
      }
      else {
         ErrorLog() << message;
      }
   }
}

std::string CheckStatsAllZeros::errorMessage(
      std::map<int, LayerStats const> const &badCounts,
      double badTime,
      std::string const &baseMessage) const {
   if (badCounts.empty()) {
      return std::string("");
   }

   std::stringstream message("");
   message << "Probe " << mParams->getName() << " has " << baseMessage
           << " at time " << badTime << "\n";
   for (auto const &b : badCounts) {
      int batchIndex          = b.first;
      LayerStats const &stats = b.second;
      message << "    batch element " << batchIndex << " has " << stats.mNumNonzero
              << " values exceeding the threshold. "
              << "Min = " << stats.mMin << "; Max = " << stats.mMax << "\n";
   }
   return message.str();
}

void CheckStatsAllZeros::ioParam_exitOnFailure(ParamsIOSwitch ioSwitch, std::shared_ptr<ParamsIO> paramsIO) {
   paramsIO->ioParam(ioSwitch, "exitOnFailure", &mExitOnFailure);
}

void CheckStatsAllZeros::ioParam_immediateExitOnFailure(ParamsIOSwitch ioSwitch, std::shared_ptr<ParamsIO> paramsIO) {
   pvAssert(!paramsIO->presentAndNotBeenRead("exitOnFailure"));
   if (mExitOnFailure) {
      paramsIO->ioParam(ioSwitch, "immediateExitOnFailure", &mImmediateExitOnFailure);
   }
   else {
      mImmediateExitOnFailure = false;
   }
}

void CheckStatsAllZeros::ioParamsFillGroup(ParamsIOSwitch ioSwitch, std::shared_ptr<ParamsIO> paramsIO) {
   ioParam_exitOnFailure(ioSwitch, paramsIO);
   ioParam_immediateExitOnFailure(ioSwitch, paramsIO);
}

void CheckStatsAllZeros::setFirstFailure(
      std::map<int, LayerStats const> const &failureMap,
      double failureTime) {
   if (!foundNonzero()) {
      mFirstFailureTime = failureTime;
      for (auto &p : failureMap) {
         mFirstFailure.insert(p);;
      }
   }
}

} // namespace PV
