/*
 * RunningAverageLayer.hpp
 * This layer stores the running moving average of the most recent n flips (updates) of a layer.
 *
 *  Created on: Mar 3, 2015
 *      Author: wchavez
 */

// RunningAverageLayer was deprecated on Aug 15, 2018.

#ifndef RUNNINGAVERAGELAYER_HPP_
#define RUNNINGAVERAGELAYER_HPP_

#include "layers/CloneVLayer.hpp"

namespace PV {

class RunningAverageLayer : public CloneVLayer {
  public:
   RunningAverageLayer(const char *name, PVParams *params, Communicator const *comm);
   virtual ~RunningAverageLayer();
   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   virtual Response::Status updateState(double timef, double dt) override;
   virtual int setActivity() override;
   int getNumImagesToAverage() { return numImagesToAverage; }

  protected:
   RunningAverageLayer();
   void initialize(const char *name, PVParams *params, Communicator const *comm);
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   void ioParam_numImagesToAverage(enum ParamsIOFlag ioFlag);

  private:
   int initialize_base();

  protected:
   int numImagesToAverage;
   int numUpdateTimes;
}; // class RunningAverageLayer

} // namespace PV

#endif /* CLONELAYER_HPP_ */
