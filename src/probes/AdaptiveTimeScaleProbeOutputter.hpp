#ifndef ADAPTIVETIMESCALEPROBEOUTPUTTER_HPP_
#define ADAPTIVETIMESCALEPROBEOUTPUTTER_HPP_

#include "columns/Communicator.hpp"
#include "components/AdaptiveTimeScaleController.hpp" // TimeScaleData
#include "io/PrintStream.hpp"
#include "probes/BaseProbeOutputter.hpp"
#include "probes/ProbeDataBuffer.hpp"

#include <memory>

namespace PV {

class AdaptiveTimeScaleProbeOutputter : public BaseProbeOutputter {
  protected:
   /**
    * List of parameters needed from the AdaptiveTimeScaleProbeOutputter class
    * @name AdaptiveTimeScaleProbeOutputter Parameters
    * @{
    */

   /**
    * @brief writeTimeScaleFieldnames: A flag to determine if fieldnames are
    * written to the HyPerCol_timescales file. If false, file is written as
    * comma a separated list. Default is true.
    */
   virtual void ioParam_writeTimeScaleFieldnames(ParamsIOSwitch ioSwitch);
   /** @} */

  public:
   AdaptiveTimeScaleProbeOutputter(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);
   virtual ~AdaptiveTimeScaleProbeOutputter() {}

   virtual void ioParamsFillGroup(ParamsIOSwitch ioSwitch);

   void printTimeScaleBuffer(ProbeDataBuffer<TimeScaleData> const &storedValues);

  protected:
   AdaptiveTimeScaleProbeOutputter() {}
   void initialize(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);

   void printTimeScaleData(
         std::shared_ptr<PrintStream> printStreamPtr,
         double timestamp,
         int batchIndex,
         TimeScaleData const &timeScaleData);
   void printToFiles(ProbeDataBuffer<TimeScaleData> const &storedValues);
   void printToLog(ProbeDataBuffer<TimeScaleData> const &storedValues);

  protected:
   bool mWriteTimeScaleFieldnames = true;
}; // class AdaptiveTimeScaleProbeOutputter

} // namespace PV

#endif // ADAPTIVETIMESCALEPROBEOUTPUTTER_HPP_
