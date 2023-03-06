#include "CheckStatsAllZeros.hpp"

#include "utils/PVAssert.hpp"
#include "utils/PVLog.hpp"

#include <sstream>
#include <utility>

namespace PV {

CheckStatsAllZeros::CheckStatsAllZeros(char const *objName, PVParams *params)
      : mName(objName), mParams(params) {}

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
      pvAssert(!mExitOnFailure);
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
   message << "Probe " << mName.c_str() << " has " << baseMessage << " at time " << badTime << "\n";
   for (auto const &b : badCounts) {
      int batchIndex          = b.first;
      LayerStats const &stats = b.second;
      message << "    batch element " << batchIndex << " has " << stats.mNumNonzero
              << " values exceeding the threshold. "
              << "Min = " << stats.mMin << "; Max = " << stats.mMax << "\n";
   }
   return message.str();
}

void CheckStatsAllZeros::ioParam_exitOnFailure(enum ParamsIOFlag ioFlag) {
   mParams->ioParamValue(ioFlag, mName.c_str(), "exitOnFailure", &mExitOnFailure, mExitOnFailure);
}

void CheckStatsAllZeros::ioParam_immediateExitOnFailure(enum ParamsIOFlag ioFlag) {
   pvAssert(!mParams->presentAndNotBeenRead(mName.c_str(), "exitOnFailure"));
   if (mExitOnFailure) {
      mParams->ioParamValue(
            ioFlag,
            mName.c_str(),
            "immediateExitOnFailure",
            &mImmediateExitOnFailure,
            mImmediateExitOnFailure);
   }
   else {
      mImmediateExitOnFailure = false;
   }
}

void CheckStatsAllZeros::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_exitOnFailure(ioFlag);
   ioParam_immediateExitOnFailure(ioFlag);
}

void CheckStatsAllZeros::setFirstFailure(
      std::map<int, LayerStats const> const &failureMap,
      double failureTime) {
   if (!foundNonzero()) {
      mFirstFailureTime = failureTime;
      mFirstFailure     = failureMap;
   }
}

} // namespace PV
