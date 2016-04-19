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
   virtual int communicateInitInfo();

   virtual int outputState(double timed);

protected:
   int initRescaleLayerTestProbe(const char * probeName, HyPerCol * hc);
   void ioParam_buffer(enum ParamsIOFlag ioFlag);
   bool colinear(int nx, int ny, int ystrideA, int ystrideB, pvadata_t const * A, pvadata_t const * B, double tolerance, double * cov, double * stdA, double * stdB);

private:
   int initRescaleLayerTestProbe_base();

}; // end class RescaleLayerTestProbe

BaseObject * createRescaleLayerTestProbe(char const * name, HyPerCol * hc);

}  // end namespace PV
#endif /* RESCALELAYERTESTPROBE_HPP_ */
