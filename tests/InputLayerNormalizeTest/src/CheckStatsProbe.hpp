/*
 * CheckStatsProbe.hpp
 *
 *  Created on: May 3, 2017
 *      Author: peteschultz
 */

#ifndef CHECKSTATSPROBE_HPP_
#define CHECKSTATSPROBE_HPP_

#include <columns/Communicator.hpp>
#include <params/PVParams.hpp>
#include <probes/StatsProbeImmediate.hpp>

class CheckStatsProbe : public PV::StatsProbeImmediate {
  protected:
   virtual void ioParam_correctMin(PV::ParamsIOSwitch ioSwitch);
   virtual void ioParam_correctMax(PV::ParamsIOSwitch ioSwitch);
   virtual void ioParam_correctMean(PV::ParamsIOSwitch ioSwitch);
   virtual void ioParam_correctStd(PV::ParamsIOSwitch ioSwitch);
   virtual void ioParam_tolerance(PV::ParamsIOSwitch ioSwitch);

  public:
   CheckStatsProbe(std::shared_ptr<PV::ParamsIO> paramsIO, PV::Communicator const *comm);
   virtual ~CheckStatsProbe();

  protected:
   CheckStatsProbe();
   virtual void checkStats() override;
   void initialize(std::shared_ptr<PV::ParamsIO> paramsIO, PV::Communicator const *comm);
   virtual int ioParamsFillGroup(PV::ParamsIOSwitch ioSwitch) override;

  protected:
   // Defaults are taken from U(0,1).
   float mCorrectMin  = 0.0f;
   float mCorrectMax  = 1.0f;
   float mCorrectMean = 0.5f;
   float mCorrectStd  = 0.28867513f;
   float mTolerance   = 0.000001f;
};

#endif // CHECKSTATSPROBE_HPP_
