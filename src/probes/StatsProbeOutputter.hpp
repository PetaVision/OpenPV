#ifndef STATSPROBEOUTPUTTER_HPP_
#define STATSPROBEOUTPUTTER_HPP_

#include "columns/Communicator.hpp"
#include "io/PVParams.hpp"
#include "io/PrintStream.hpp"
#include "probes/BaseProbeOutputter.hpp"
#include "probes/ProbeDataBuffer.hpp"
#include "probes/StatsProbeTypes.hpp"

#include <memory>

namespace PV {

class StatsProbeOutputter : public BaseProbeOutputter {
  public:
   StatsProbeOutputter(char const *objName, PVParams *params, Communicator const *comm);
   virtual ~StatsProbeOutputter() {}

   void printGlobalStatsBuffer(ProbeDataBuffer<LayerStats> const &storedValues);

   void setConvertToHertzFlag(bool convertToHertzFlag);

  protected:
   StatsProbeOutputter() {}
   void initialize(char const *objName, PVParams *params, Communicator const *comm);

   void printLayerStats(
         std::shared_ptr<PrintStream> printStreamPtr,
         LayerStats const &stats,
         double timestamp,
         int batchIndex);
   void printToFiles(ProbeDataBuffer<LayerStats> const &storedValues);
   void printToLog(ProbeDataBuffer<LayerStats> const &storedValues);

  private:
   bool mConvertToHertz = false; // If true, multiply average by 1000 in the output and label as Hz.

}; // class StatsProbeOutputter

} // namespace PV

#endif // STATSPROBEOUTPUTTER_HPP_
