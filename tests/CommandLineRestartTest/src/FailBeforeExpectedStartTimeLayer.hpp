#ifndef FAILBEFOREEXPECTEDSTARTTIMELAYER_HPP_
#define FAILBEFOREEXPECTEDSTARTTIMELAYER_HPP_

#include "layers/HyPerLayer.hpp"

/**
 * A class to make sure that the column does not call LayerUpdateState before the expected
 * starting time. The use case is for testing that restarting from checkpoint does set the
 * simulation time the way it should.
 * Note that the data member is initialized to +infinity, so that the layer will always
 * fail unless the expected start time is set by calling setExpectedStartTime before
 * LayerUpdateState is called.
 */
class FailBeforeExpectedStartTimeLayer : public PV::HyPerLayer {
  public:
   FailBeforeExpectedStartTimeLayer(
         char const *name,
         PV::PVParams *params,
         PV::Communicator const *comm);
   ~FailBeforeExpectedStartTimeLayer() {}

   void setExpectedStartTime(double expectedStartTime) { mExpectedStartTime = expectedStartTime; }

  protected:
   FailBeforeExpectedStartTimeLayer();
   void initialize(char const *name, PV::PVParams *params, PV::Communicator const *comm);
   virtual PV::Response::Status checkUpdateState(double simTime, double dt) override;

  private:
   double mExpectedStartTime = std::numeric_limits<double>::infinity();
};

#endif // FAILBEFOREEXPECTEDSTARTTIMELAYER_HPP_
