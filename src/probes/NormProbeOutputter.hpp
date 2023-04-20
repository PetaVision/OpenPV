#ifndef L1NORMPROBEOUTPUTTER_HPP_
#define L1NORMPROBEOUTPUTTER_HPP_

#include "columns/Communicator.hpp"
#include "io/PVParams.hpp"
#include "io/PrintStream.hpp"
#include "probes/BaseProbeOutputter.hpp"
#include "probes/ProbeDataBuffer.hpp"

#include <memory>

namespace PV {

class NormProbeOutputter : public BaseProbeOutputter {
  public:
   NormProbeOutputter(char const *objName, PVParams *params, Communicator const *comm);
   virtual ~NormProbeOutputter() {}

   void printGlobalNormsBuffer(ProbeDataBuffer<double> const &storedValues, int numNeurons);

  protected:
   NormProbeOutputter() {}
   void initialize(char const *objName, PVParams *params, Communicator const *comm);

   void printNorm(
         std::shared_ptr<PrintStream> printStreamPtr,
         double timestamp,
         int numNeurons,
         int batchIndex,
         double norm);
   void printToFiles(ProbeDataBuffer<double> const &storedValues, int numNeurons);
   void printToLog(ProbeDataBuffer<double> const &storedValues, int numNeurons);

}; // class NormProbeOutputter

} // namespace PV

#endif // L1NORMPROBEOUTPUTTER_HPP_
