/*
 * RescaleLayerTestProbe.hpp
 *
 *  Created on: Sep 1, 2011
 *      Author: gkenyon
 */

#ifndef RESCALELAYERTESTPROBE_HPP_
#define RESCALELAYERTESTPROBE_HPP_

#include <io/StatsProbe.hpp>

namespace PV {

class RescaleLayerTestProbe: public PV::StatsProbe {
public:
   RescaleLayerTestProbe(const char * probeName, HyPerCol * hc);

   virtual int outputState(double timed);

protected:
   int initRescaleLayerTestProbe(const char * probeName, HyPerCol * hc);
   void ioParam_buffer(enum ParamsIOFlag ioFlag);

private:
   int initRescaleLayerTestProbe_base();

};

} /* namespace PV */
#endif /* RESCALELAYERTESTPROBE_HPP_ */
