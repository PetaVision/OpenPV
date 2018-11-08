/*
 * BinningTestProbe.hpp
 *
 *  Created on: Jan 15, 2015
 *      Author: slundquist
 */

#ifndef BINNINGTESTPROBE_HPP_
#define BINNINGTESTPROBE_HPP_

#include <layers/BinningLayer.hpp>
#include <probes/LayerProbe.hpp>

namespace PV {

class BinningTestProbe : public PV::LayerProbe {
  public:
   BinningTestProbe(const char *name, PVParams *params, Communicator *comm);
   virtual void calcValues(double timeValue) {}

  protected:
   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual Response::Status outputState(double simTime, double deltaTime) override;

  private:
   BinningLayer *mBinningLayer = nullptr;
};

} /* namespace PV */
#endif // BINNINGTESTPROBE_HPP_
