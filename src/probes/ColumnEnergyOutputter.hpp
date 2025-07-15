#ifndef COLUMNENERGYOUTPUTTER_HPP_
#define COLUMNENERGYOUTPUTTER_HPP_

#include "probes/BaseProbeOutputter.hpp"
#include "probes/ProbeDataBuffer.hpp"

namespace PV {

class ColumnEnergyOutputter : public BaseProbeOutputter {
  public:
   ColumnEnergyOutputter(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);
   virtual ~ColumnEnergyOutputter() {}

   void printColumnEnergiesBuffer(ProbeDataBuffer<double> const &storedValues);

   virtual void printHeader() override;

  protected:
   ColumnEnergyOutputter() {}
   void initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);

   void printEnergy(
         std::shared_ptr<PrintStream> printStreamPtr,
         double timestamp,
         int globalBatchIndex,
         double energy);
   void printToFiles(ProbeDataBuffer<double> const &storedValues);
   void printToLog(ProbeDataBuffer<double> const &storedValues);

}; // class ColumnEnergyOutputter

} // namespace PV

#endif // COLUMNENERGYOUTPUTTER_HPP_
