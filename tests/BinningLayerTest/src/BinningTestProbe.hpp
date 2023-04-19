/*
 * BinningTestProbe.hpp
 *
 *  Created on: Jan 15, 2015
 *      Author: slundquist
 */

#ifndef BINNINGTESTPROBE_HPP_
#define BINNINGTESTPROBE_HPP_

#include <columns/BaseObject.hpp>
#include <columns/Communicator.hpp>
#include <columns/Messages.hpp>
#include <io/PVParams.hpp>
#include <layers/BinningLayer.hpp>
#include <observerpattern/Response.hpp>
#include <probes/TargetLayerComponent.hpp>

#include <memory>

namespace PV {

class BinningTestProbe : public PV::BaseObject {
  public:
   BinningTestProbe(const char *name, PVParams *params, Communicator const *comm);

  protected:
   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   void initialize(const char *name, PVParams *params, Communicator const *comm);
   virtual void initMessageActionMap() override;
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   Response::Status outputState(std::shared_ptr<LayerOutputStateMessage const> message);

   Response::Status respondLayerOutputState(std::shared_ptr<LayerOutputStateMessage const> message);
   Response::Status respondProbeWriteParams(std::shared_ptr<ProbeWriteParamsMessage const> message);

   BinningLayer *getBinningLayer() const { return mBinningLayer; }

  private:
   BinningLayer *mBinningLayer = nullptr;
   std::shared_ptr<TargetLayerComponent> mProbeTargetLayerLocator;
};

} /* namespace PV */
#endif // BINNINGTESTPROBE_HPP_
