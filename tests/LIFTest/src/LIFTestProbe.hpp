/*
 * LIFTestProbe.hpp
 *
 *  Created on: Aug 27, 2012
 *      Author: pschultz
 */

#ifndef LIFTESTPROBE_HPP_
#define LIFTESTPROBE_HPP_

#include "checkpointing/Checkpointer.hpp"
#include "checkpointing/CheckpointingMessages.hpp"
#include "columns/Communicator.hpp"
#include "columns/Messages.hpp"
#include "params/PVParams.hpp"
#include "observerpattern/Response.hpp"
#include "probes/StatsProbeImmediate.hpp"
#include <memory>
#include <vector>

namespace PV {

class LIFTestProbe : public StatsProbeImmediate {
  public:
   LIFTestProbe(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);
   virtual ~LIFTestProbe();

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

  protected:
   LIFTestProbe();
   void checkStats() override;
   void initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);
   virtual int ioParamsFillGroup(ParamsIOSwitch ioSwitch) override;
   virtual void ioParam_endingTime(ParamsIOSwitch ioSwitch);
   virtual void ioParam_tolerance(ParamsIOSwitch ioSwitch);

   Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;

  private:
   std::vector<double> mRadii;
   std::vector<double> mRates;
   std::vector<double> mTargetRates;
   std::vector<double> mStdDevs;
   std::vector<int> mCounts;

   double mEndingTime = 2000.0; // Default stop time
   double mTolerance  = 3.0; // Number of standard deviations that the observed rates can differ
   // from the expected rates.

   static constexpr const int mNumBins = 5;
};

} /* namespace PV */
#endif /* LIFTESTPROBE_HPP_ */
