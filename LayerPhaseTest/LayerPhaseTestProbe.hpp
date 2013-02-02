/*
 * LayerPhaseTestProbe.hpp
 *
 *  Created on: January 27, 2013
 *      Author: garkenyon
 */

#ifndef LAYERPHASETESTPROBE_HPP_
#define LAYERPHASETESTPROBE_HPP_

#include <src/include/pv_arch.h>
#include <src/io/StatsProbe.hpp>
#include <src/layers/HyPerLayer.hpp>
#include <assert.h>

namespace PV {

class LayerPhaseTestProbe: public PV::StatsProbe {
public:
   LayerPhaseTestProbe(const char * name, const char * filename, HyPerLayer * layer, const char * msg);
   LayerPhaseTestProbe(const char * name, HyPerLayer * layer, const char * msg);

   virtual int outputState(double timed);
protected:
   pvdata_t equilibriumValue;
   double equilibriumTime;

};

} /* namespace PV */
#endif /* LAYERPHASETESTPROBE_HPP_ */
