/*
 * RescaleLayerTestProbe.hpp
 *
 *  Created on: Sep 1, 2011
 *      Author: gkenyon
 */

#ifndef RESCALELAYERTESTPROBE_HPP_
#define RESCALELAYERTESTPROBE_HPP_

#include <probes/StatsProbe.hpp>

#include <components/RescaleActivityBuffer.hpp>

namespace PV {

class RescaleLayerTestProbe : public PV::StatsProbe {
  public:
   RescaleLayerTestProbe(const char *name, PVParams *params, Communicator const *comm);
   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual Response::Status outputState(double simTime, double deltaTime) override;

  protected:
   void initialize(const char *name, PVParams *params, Communicator const *comm);
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

  protected:
   RescaleActivityBuffer *mRescaleBuffer = nullptr;

}; // end class RescaleLayerTestProbe

} // end namespace PV
#endif /* RESCALELAYERTESTPROBE_HPP_ */
