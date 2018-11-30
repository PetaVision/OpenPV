/*
 * KmeansLayer.hpp
 *
 *  Created on: Dec. 1, 2014
 *      Author: Xinhua Zhang
 */

// KmeansLayer was deprecated on Aug 15, 2018.

#ifndef KMEANSLAYER_HPP_
#define KMEANSLAYER_HPP_

#include "layers/HyPerLayer.hpp"

namespace PV {
class KmeansLayer : public HyPerLayer {
  public:
   KmeansLayer(const char *name, PVParams *params, Communicator *comm);
   virtual ~KmeansLayer();

  protected:
   KmeansLayer();
   void initialize(const char *name, PVParams *params, Communicator *comm);
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
