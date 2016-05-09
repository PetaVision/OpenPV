/*
 *  CudaTimer.hpp
 *
 *  Sheng Lundquist
 */

#ifndef CUDATIMER_HPP_
#define CUDATIMER_HPP_

#include "../../utils/Timer.hpp"
#include <stdio.h>

////////////////////////////////////////////////////////////////////////////////

namespace PVCuda{
#include <cuda_runtime_api.h>
   
   /**
    * A subclass of Timer to time Cuda gpu runs
    * Note that there are new variables here to store timing information, so the timer variables from Timer are not updated in CudaTimers
    */
   class CudaTimer : public PV::Timer {
   public:
      CudaTimer(double init_time=0.0);
      CudaTimer(const char * timermessage, double init_time=0.0);
      CudaTimer(const char * objname, const char * objtype, const char * timertype, double init_time=0.0);
      virtual ~CudaTimer();

      /**
       * A function to put an instruction on the GPU queue to start timing
       */
      virtual double start();
      /**
       * A function to put an instruction on the GPU queue to stop timing
       */
      virtual double stop();
      /**
       * A blocking function to accumulate to the final time between start and stop.
       * This function must be called after a pair of start/stops. Not doing so will clobber the previous start/stop pair
       * @return Returns the accumulated time of this timer
       */
      double accumulateTime();
      virtual int fprint_time(FILE * stream);
      /**
       * A setter function to set the stream to time
       */
      void setStream(cudaStream_t stream){this->stream = stream;}
      
   private:
      cudaEvent_t startEvent;
      cudaEvent_t stopEvent;
      float time; //TODO maybe use Timer's member variables to store the time?
      cudaStream_t stream;
   };
   
} // namespace PV

#endif /* CUDATIMER_HPP_ */
