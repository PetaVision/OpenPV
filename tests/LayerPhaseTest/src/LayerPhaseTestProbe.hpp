/*
 * LayerPhaseTestProbe.hpp
 *
 *  Created on: January 27, 2013
 *      Author: garkenyon
 */

#ifndef LAYERPHASETESTPROBE_HPP_
#define LAYERPHASETESTPROBE_HPP_

#include "include/pv_arch.h"
#include "layers/HyPerLayer.hpp"
#include "probes/StatsProbe.hpp"
#include "utils/PVLog.hpp"

namespace PV {

class LayerPhaseTestProbe : public PV::StatsProbe {
  public:
   LayerPhaseTestProbe(const char *name, HyPerCol *hc);

   virtual Response::Status outputState(double timestamp) override;

  protected:
   int initialize(const char *name, HyPerCol *hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_buffer(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_equilibriumValue(enum ParamsIOFlag ioFlag);
   virtual void ioParam_equilibriumTime(enum ParamsIOFlag ioFlag);

  private:
   int initialize_base();

  protected:
   float equilibriumValue;
   double equilibriumTime;
};

} /* namespace PV */
#endif /* LAYERPHASETESTPROBE_HPP_ */
