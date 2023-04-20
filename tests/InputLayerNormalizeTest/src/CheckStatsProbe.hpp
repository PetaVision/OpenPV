/*
 * CheckStatsProbe.hpp
 *
 *  Created on: May 3, 2017
 *      Author: peteschultz
 */

#ifndef CHECKSTATSPROBE_HPP_
#define CHECKSTATSPROBE_HPP_

#include <columns/Communicator.hpp>
#include <io/PVParams.hpp>
#include <probes/StatsProbeImmediate.hpp>

class CheckStatsProbe : public PV::StatsProbeImmediate {
  protected:
   virtual void ioParam_correctMin(enum PV::ParamsIOFlag ioFlag);
   virtual void ioParam_correctMax(enum PV::ParamsIOFlag ioFlag);
   virtual void ioParam_correctMean(enum PV::ParamsIOFlag ioFlag);
   virtual void ioParam_correctStd(enum PV::ParamsIOFlag ioFlag);
   virtual void ioParam_tolerance(enum PV::ParamsIOFlag ioFlag);

  public:
   CheckStatsProbe(char const *name, PV::PVParams *params, PV::Communicator const *comm);
   virtual ~CheckStatsProbe();

  protected:
   CheckStatsProbe();
   virtual void checkStats() override;
   void initialize(char const *name, PV::PVParams *params, PV::Communicator const *comm);
   virtual int ioParamsFillGroup(enum PV::ParamsIOFlag ioFlag) override;

  protected:
   // Defaults are taken from U(0,1).
   float mCorrectMin  = 0.0f;
   float mCorrectMax  = 1.0f;
   float mCorrectMean = 0.5f;
   float mCorrectStd  = 0.28867513f;
   float mTolerance   = 0.000001f;
};

#endif // CHECKSTATSPROBE_HPP_
