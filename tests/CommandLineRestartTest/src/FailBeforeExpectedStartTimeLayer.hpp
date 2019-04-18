#ifndef FAILBEFOREEXPECTEDSTARTTIMELAYER_HPP_
#define FAILBEFOREEXPECTEDSTARTTIMELAYER_HPP_

#include "layers/HyPerLayer.hpp"

/**
 * A class to make sure that the column does not call updateState before the expected
 * starting time. The use case is for testing that restarting from checkpoint does
 * set the simulation time the way it should.
 * Note that the data member is initialized to +infinity, so that the layer
 * will always fail unless the expected start time is set by calling
 * setExpectedStartTime before updateState is called.
 */
class FailBeforeExpectedStartTimeLayer : public PV::HyPerLayer {
  public:
   FailBeforeExpectedStartTimeLayer(char const *name, PV::HyPerCol *hc);
   ~FailBeforeExpectedStartTimeLayer() {}

   void setExpectedStartTime(double expectedStartTime) { mExpectedStartTime = expectedStartTime; }

  protected:
   FailBeforeExpectedStartTimeLayer();
   int initialize(char const *name, PV::HyPerCol *hc);
#ifdef PV_USE_CUDA
   virtual PV::Response::Status updateStateGpu(double simTime, double dt) override;
#endif
   virtual PV::Response::Status updateState(double simTime, double dt) override;

  private:
   int initialize_base();

  private:
   double mExpectedStartTime = std::numeric_limits<double>::infinity();
};

#endif // FAILBEFOREEXPECTEDSTARTTIMELAYER_HPP_
