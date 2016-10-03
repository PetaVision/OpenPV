/*
 *  CudaTimer.cu
 *
 *  Sheng Lundquist
 */

#include "CudaTimer.hpp"
#include "../../io/PrintStream.hpp"
#include "cuda_util.hpp"

namespace PVCuda {

CudaTimer::CudaTimer(double init_time) : PV::Timer(init_time) {
   handleError(cudaEventCreate(&startEvent), "Start event creation");
   handleError(cudaEventCreate(&stopEvent), "Stop event creation");
   time   = 0;
   stream = nullptr;
}

CudaTimer::CudaTimer(const char *timermessage, double init_time)
      : PV::Timer(timermessage, init_time) {
   handleError(cudaEventCreate(&startEvent), "Start event creation");
   handleError(cudaEventCreate(&stopEvent), "Stop event creation");
   time   = 0;
   stream = nullptr;
}

CudaTimer::CudaTimer(
      const char *objname,
      const char *objtype,
      const char *timertype,
      double init_time)
      : PV::Timer(objname, objtype, timertype, init_time) {
   handleError(cudaEventCreate(&startEvent), "Start event creation");
   handleError(cudaEventCreate(&stopEvent), "Stop event creation");
   time   = 0;
   stream = nullptr;
}

CudaTimer::~CudaTimer() {
   handleError(cudaEventDestroy(startEvent), "Start event destruction");
   handleError(cudaEventDestroy(stopEvent), "Stop event destruction");
}

double CudaTimer::start() {
   handleError(cudaEventRecord(startEvent, stream), "Recording start event");
   return 0;
}

double CudaTimer::stop() {
   handleError(cudaEventRecord(stopEvent, stream), "Recording stop event");
   return 0;
}

// Note this function is blocking
double CudaTimer::accumulateTime() {
   float curTime;
   handleError(cudaEventSynchronize(stopEvent), "Synchronizing stop event");
   handleError(cudaEventElapsedTime(&curTime, startEvent, stopEvent), "Calculating elapsed time");
   // Roundoff errors?
   time += curTime;
   return (double)time;
}

int CudaTimer::fprint_time(PrintStream &stream) {
   if (rank == 0) {
      stream << message << "processor cycle time == " << time << std::endl;
   }
   return 0;
}

} // namespace PV
