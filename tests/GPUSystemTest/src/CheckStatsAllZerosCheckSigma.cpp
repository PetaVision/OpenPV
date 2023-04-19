#include "CheckStatsAllZerosCheckSigma.hpp"
#include "utils/PVLog.hpp"
#include <string>

namespace PV {

CheckStatsAllZerosCheckSigma::CheckStatsAllZerosCheckSigma(char const *objName, PVParams *params)
      : CheckStatsAllZeros(objName, params) {}

CheckStatsAllZerosCheckSigma::~CheckStatsAllZerosCheckSigma() {}

std::map<int, LayerStats const>
CheckStatsAllZerosCheckSigma::checkStats(ProbeData<LayerStats> const &statsBatch) {
   double const tolSigma = 5e-5;
   int nbatch            = static_cast<int>(statsBatch.size());
   std::map<int, LayerStats const> result;
   bool badSigma = false;
   for (int b = 0; b < nbatch; ++b) {
      auto &stats = statsBatch.getValue(b);
      bool bad    = false;
      if (stats.mNumNonzero != 0) {
         bad = true;
      }
      double sigma = stats.sigma();
      if (sigma > tolSigma) {
         badSigma = true;
         bad      = true;
      }
      if (bad) {
         result.emplace_hint(result.end(), b, stats);
      }
   }
   if (!result.empty()) {
      if (!foundNonzero()) {
         setFirstFailure(result, statsBatch.getTimestamp());
      }
      auto message = errorMessage(result, statsBatch.getTimestamp(), "nonzero activity");
      if (badSigma) {
         message.append("Probe \"")
               .append(getName())
               .append("\": Nonzero standard deviation at time ");
         message.append(std::to_string(statsBatch.getTimestamp()));
         message.append("; tolerance is ").append(std::to_string(tolSigma)).append("\n");
      }
      if (getImmediateExitOnFailure()) {
         Fatal() << message;
      }
      else {
         ErrorLog() << message;
      }
   }
   return result;
}

} // namespace PV
