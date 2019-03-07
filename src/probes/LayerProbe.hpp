/*
 * LayerProbe.h
 *
 *  Created on: Mar 7, 2009
 *      Author: rasmussn
 */

#ifndef LAYERPROBE_HPP_
#define LAYERPROBE_HPP_

#include "BaseProbe.hpp"
#include "io/fileio.hpp"
#include "layers/HyPerLayer.hpp"
#include "utils/Timer.hpp"
#include <stdio.h>

namespace PV {

typedef enum { BufV, BufActivity } PVBufType;

/**
 * The base class for probes attached to layers.
 */
class LayerProbe : public BaseProbe {

   // Methods
  protected:
   /**
    * List of parameters for the LayerProbe class
    * @name LayerProbe Parameters
    * @{
    */

   /**
    * @brief targetName: the name of the layer to attach the probe to.
    * In LayerProbes, targetLayer can be used in the params file instead of
    * targetName.  LayerProbe looks for targetLayer first and then targetName.
    */
   virtual void ioParam_targetName(enum ParamsIOFlag ioFlag) override;
   /** @} */
  public:
   LayerProbe(const char *name, PVParams *params, Communicator *comm);
   virtual ~LayerProbe();

   /**
    * Called by HyPerCol::run.  It calls BaseProbe::communicateInitInfo, then
    * checks that the targetLayer/targetName parameter refers to a HyPerLayer.
    */
   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   Response::Status
   respondLayerProbeWriteParams(std::shared_ptr<LayerProbeWriteParamsMessage const> message);
   Response::Status respondLayerOutputState(std::shared_ptr<LayerOutputStateMessage const> message);

   virtual Response::Status outputStateWrapper(double simTime, double deltaTime) override;

   HyPerLayer *getTargetLayer() { return targetLayer; }

  protected:
   LayerProbe();
   void initialize(const char *name, PVParams *params, Communicator *comm);
   virtual void initMessageActionMap() override;

   virtual Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;

   /**
    * Implements the needRecalc method.  Returns true if the target layer's
    * getLastUpdateTime method
    * is greater than the probe's lastUpdateTime member variable.
    */
   virtual bool needRecalc(double timevalue) override;

   /**
    * Implements the referenceUpdateTime method.  Returns the last update time of
    * the target layer.
    */
   virtual double referenceUpdateTime(double simTime) const override;

  private:
   int initialize_base();

   // Member variables
  protected:
   HyPerLayer *targetLayer = nullptr;
   Timer *mIOTimer         = nullptr;
};
}

#endif /* LAYERPROBE_HPP_ */
