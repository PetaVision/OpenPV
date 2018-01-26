/*
 * KmeansLayer.hpp
 *
 *  Created on: Dec. 1, 2014
 *      Author: Xinhua Zhang
 */

#ifndef KMEANSLAYER_HPP_
#define KMEANSLAYER_HPP_

#include "HyPerLayer.hpp"

namespace PV {
class KmeansLayer : public HyPerLayer {
  public:
   KmeansLayer(const char *name, HyPerCol *hc);
   virtual bool activityIsSpiking() override { return false; }
   virtual ~KmeansLayer();

  protected:
   KmeansLayer();
   int initialize(const char *name, HyPerCol *hc);
   virtual Response::Status updateState(double time, double dt) override;
   virtual int setActivity() override;
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_TrainingFlag(enum ParamsIOFlag ioFlag);
   bool trainingFlag;

  private:
   int initialize_base();

}; // class KmeansLayer

} // namespace PV

#endif /* KMEANSLAYER_HPP_ */
