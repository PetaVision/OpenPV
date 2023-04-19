/*
 * LayerPhaseTestProbe.hpp
 *
 *  Created on: January 27, 2013
 *      Author: garkenyon
 */

#ifndef LAYERPHASETESTPROBE_HPP_
#define LAYERPHASETESTPROBE_HPP_

#include "columns/Communicator.hpp"
#include "io/PVParams.hpp"
#include "probes/StatsProbeImmediate.hpp"

namespace PV {

class LayerPhaseTestProbe : public PV::StatsProbeImmediate {
  public:
   LayerPhaseTestProbe(const char *name, PVParams *params, Communicator const *comm);

  protected:
   virtual void checkStats() override;
   virtual void createProbeLocal(char const *name, PVParams *params) override;
   void initialize(const char *name, PVParams *params, Communicator const *comm);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_equilibriumValue(enum ParamsIOFlag ioFlag);
   virtual void ioParam_equilibriumTime(enum ParamsIOFlag ioFlag);

  protected:
   float mEquilibriumValue = 0.0f;
   double mEquilibriumTime = 0.0;
};

} /* namespace PV */
#endif /* LAYERPHASETESTPROBE_HPP_ */
