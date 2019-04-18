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
   RescaleLayerTestProbe(const char *name, HyPerCol *hc);
   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual Response::Status outputState(double timestamp) override;

  protected:
   int initialize(const char *name, HyPerCol *hc);
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
   int initialize_base();

}; // end class RescaleLayerTestProbe

} // end namespace PV
#endif /* RESCALELAYERTESTPROBE_HPP_ */
