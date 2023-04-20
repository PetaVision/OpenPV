#ifndef QUOTIENTPROBEOUTPUTTER_HPP_
#define QUOTIENTPROBEOUTPUTTER_HPP_

#include "probes/BaseProbeOutputter.hpp"
#include "probes/ProbeDataBuffer.hpp"

namespace PV {

class QuotientProbeOutputter : public BaseProbeOutputter {
  public:
   QuotientProbeOutputter(char const *objName, PVParams *params, Communicator const *comm);
   virtual ~QuotientProbeOutputter() {}

   void printBuffer(ProbeDataBuffer<double> const &storedValues);

   virtual void printHeader() override;

  protected:
   QuotientProbeOutputter() {}
   void initialize(char const *objName, PVParams *params, Communicator const *comm);

   void
   print(std::shared_ptr<PrintStream> printStreamPtr,
         double timestamp,
         int globalBatchIndex,
         double energy);
   void printToFiles(ProbeDataBuffer<double> const &storedValues);
   void printToLog(ProbeDataBuffer<double> const &storedValues);

}; // class QuotientProbeOutputter

} // namespace PV

#endif // QUOTIENTPROBEOUTPUTTER_HPP_
