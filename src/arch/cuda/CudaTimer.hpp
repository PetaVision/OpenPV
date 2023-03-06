/*
 *  CudaTimer.hpp
 *
 *  Sheng Lundquist
 */

#ifndef CUDATIMER_HPP_
#define CUDATIMER_HPP_

#include "io/PrintStream.hpp"
#include "utils/Timer.hpp"
#include <cuda_runtime_api.h>

////////////////////////////////////////////////////////////////////////////////

namespace PVCuda {

/**
 * A subclass of Timer to time Cuda gpu runs
 * Note that there are new variables here to store timing information, so the timer variables from
 * Timer are not updated in CudaTimers
 */
class CudaTimer : public PV::Timer {
  public:
   CudaTimer(const char *timermessage, double init_time = 0.0);
   CudaTimer(
         const char *objname,
         const char *objtype,
         const char *timertype,
         double init_time = 0.0);
   virtual ~CudaTimer();

   /**
    * A function to put an instruction on the GPU queue to start timing
    */
   virtual double start() override;
   /**
    * A function to put an instruction on the GPU queue to stop timing.
    * Internally sets the mEventPending flag, to signal to accumulateTime()
    * to add the time of the start/stop pair.
    */
   virtual double stop() override;
   /**
    * A blocking function to accumulate to the final time between start and stop.
    * This function must be called after a pair of start/stops. Not doing so will clobber the
    * previous start/stop pair. If the mEventPending flag is true, adds the time of the pending
    * start/stop pair and clears the flag. If mEventPending is false, this routine does nothing.
    * @return Returns the accumulated time of this timer
    */
   double accumulateTime();
   virtual int fprint_time(PV::PrintStream &printStream) const override;
   /**
    * A setter function to set the stream to time
    */
   void setStream(cudaStream_t stream) { mStream = stream; }

  private:
   cudaEvent_t mStartEvent;
   cudaEvent_t mStopEvent;
   float mTime; // TODO maybe use Timer's member variables to store the time?
   cudaStream_t mStream;
   bool mEventPending = false; // A flag that is set by stop(). If true, accumulateTime() will add
};

} // namespace PVCuda

#endif /* CUDATIMER_HPP_ */
