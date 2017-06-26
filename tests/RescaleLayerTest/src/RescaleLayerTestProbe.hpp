/*
 * RescaleLayerTestProbe.hpp
 *
 *  Created on: Sep 1, 2011
 *      Author: gkenyon
 */

#ifndef RESCALELAYERTESTPROBE_HPP_
#define RESCALELAYERTESTPROBE_HPP_

#include "probes/StatsProbe.hpp"

namespace PV {

class RescaleLayerTestProbe : public PV::StatsProbe {
  public:
   RescaleLayerTestProbe(const char *probeName, HyPerCol *hc);
   virtual int communicateInitInfo(CommunicateInitInfoMessage const *message) override;

   virtual int outputState(double timed) override;

  protected:
   int initRescaleLayerTestProbe(const char *probeName, HyPerCol *hc);
   void ioParam_buffer(enum ParamsIOFlag ioFlag) override;
   bool colinear(
         int nx,
         int ny,
         int ystrideA,
         int ystrideB,
         float const *A,
         float const *B,
         double tolerance,
         double *cov,
         double *stdA,
         double *stdB);

  private:
   int initRescaleLayerTestProbe_base();

}; // end class RescaleLayerTestProbe

} // end namespace PV
#endif /* RESCALELAYERTESTPROBE_HPP_ */
