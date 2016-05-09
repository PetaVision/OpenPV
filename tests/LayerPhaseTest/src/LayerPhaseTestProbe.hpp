/*
 * LayerPhaseTestProbe.hpp
 *
 *  Created on: January 27, 2013
 *      Author: garkenyon
 */

#ifndef LAYERPHASETESTPROBE_HPP_
#define LAYERPHASETESTPROBE_HPP_

#include <include/pv_arch.h>
#include <io/StatsProbe.hpp>
#include <layers/HyPerLayer.hpp>
#include <assert.h>

namespace PV {

class LayerPhaseTestProbe: public PV::StatsProbe {
public:
   LayerPhaseTestProbe(const char * probeName, HyPerCol * hc);

   virtual int outputState(double timed);

protected:
   int initLayerPhaseTestProbe(const char * probeName, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_buffer(enum ParamsIOFlag ioFlag);
   virtual void ioParam_equilibriumValue(enum ParamsIOFlag ioFlag);
   virtual void ioParam_equilibriumTime(enum ParamsIOFlag ioFlag);

private:
   int initLayerPhaseTestProbe_base();

protected:
   pvdata_t equilibriumValue;
   double equilibriumTime;

};

BaseObject * createLayerPhaseTestProbe(char const * probeName, HyPerCol * hc);

} /* namespace PV */
#endif /* LAYERPHASETESTPROBE_HPP_ */
