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
   
   class CudaTimer : public PV::Timer {
   public:
      CudaTimer(double init_time=0.0);
      CudaTimer(const char * timermessage, double init_time=0.0);
      CudaTimer(const char * objname, const char * objtype, const char * timertype, double init_time=0.0);
      virtual ~CudaTimer();

      virtual double start();
      virtual double stop();
      double accumulateTime();
      virtual int fprint_time(FILE * stream);
      void setStream(cudaStream_t stream){this->stream = stream;}
      
   private:
      cudaEvent_t startEvent;
      cudaEvent_t stopEvent;
      float time; //TODO maybe use Timer's member variables to store the time?
      cudaStream_t stream;
   };
   
} // namespace PV

#endif /* CUDATIMER_HPP_ */
