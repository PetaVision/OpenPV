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
#include <stdio.h>

namespace PV {

class HyPerCol;
class HyPerLayer;

typedef enum { BufV, BufActivity } PVBufType;

/**
 * The base class for probes attached to layers.
 */
class LayerProbe : public BaseProbe {

   // Methods
  public:
   LayerProbe(const char *name, HyPerCol *hc);
   virtual ~LayerProbe();

   /**
    * Called by HyPerCol::run.  It calls BaseProbe::communicateInitInfo, then
    * checks that
    * the targetLayer/targetName parameter refers to a HyPerLayer in the parent
    * HyPerCol,
    * and then calls the layer's insertProbe method.
    */
   virtual int
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   HyPerLayer *getTargetLayer() { return targetLayer; }

  protected:
   LayerProbe();
   int initialize(const char *name, HyPerCol *hc);

   /**
    * List of parameters for the LayerProbe class
    * @name LayerProbe Parameters
    * @{
    */

   /**
    * @brief targetName: the name of the layer to attach the probe to.
    * In LayerProbes, targetLayer can be used in the params file instead of
    * targetName.  LayerProbe
    * looks for targetLayer first
    * and then targetName.
    */
   virtual void ioParam_targetName(enum ParamsIOFlag ioFlag) override;
   /** @} */

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
   virtual double referenceUpdateTime() const override;

  private:
   int initialize_base();

   // Member variables
  protected:
   HyPerLayer *targetLayer;
};
}

#endif /* LAYERPROBE_HPP_ */
