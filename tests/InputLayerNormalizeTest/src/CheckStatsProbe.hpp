/*
 * CheckStatsProbe.hpp
 *
 *  Created on: May 3, 2017
 *      Author: peteschultz
 */

#ifndef CHECKSTATSPROBE_HPP_
#define CHECKSTATSPROBE_HPP_

#include <probes/StatsProbe.hpp>

class CheckStatsProbe : public PV::StatsProbe {
  protected:
   virtual void ioParam_buffer(enum PV::ParamsIOFlag ioFlag) override;

   virtual void ioParam_correctMin(enum PV::ParamsIOFlag ioFlag);
   virtual void ioParam_correctMax(enum PV::ParamsIOFlag ioFlag);
   virtual void ioParam_correctMean(enum PV::ParamsIOFlag ioFlag);
   virtual void ioParam_correctStd(enum PV::ParamsIOFlag ioFlag);
   virtual void ioParam_tolerance(enum PV::ParamsIOFlag ioFlag);

  public:
   CheckStatsProbe(char const *name, PV::HyPerCol *hc);
   virtual ~CheckStatsProbe();

  protected:
   CheckStatsProbe();
   int initialize(char const *name, PV::HyPerCol *hc);
   virtual int ioParamsFillGroup(enum PV::ParamsIOFlag ioFlag) override;
   virtual PV::Response::Status outputState(double timestamp) override;

  private:
   int initialize_base();

  protected:
   // Defaults are taken from U(0,1).
   float correctMin  = 0.0f;
   float correctMax  = 1.0f;
   float correctMean = 0.5f;
   float correctStd  = 0.28867513f;
   float tolerance   = 0.000001f;
};

#endif // CHECKSTATSPROBE_HPP_
