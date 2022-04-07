/*
 *  CudaTimer.cu
 *
 *  Sheng Lundquist
 */

#include "CudaTimer.hpp"
#include "../../io/PrintStream.hpp"
#include "cuda_util.hpp"

namespace PVCuda {

CudaTimer::CudaTimer(const char *timermessage, double init_time)
      : PV::Timer(timermessage, init_time) {
   handleError(cudaEventCreate(&mStartEvent), "Start event creation");
   handleError(cudaEventCreate(&mStopEvent), "Stop event creation");
   mTime   = 0.0f;
   mStream = nullptr;
}

CudaTimer::CudaTimer(
      const char *objname,
      const char *objtype,
      const char *timertype,
      double init_time)
      : PV::Timer(objname, objtype, timertype, init_time) {
   handleError(cudaEventCreate(&mStartEvent), "Start event creation");
   handleError(cudaEventCreate(&mStopEvent), "Stop event creation");
   mTime   = 0.0f;
   mStream = nullptr;
}

CudaTimer::~CudaTimer() {
   handleError(cudaEventDestroy(mStartEvent), "Start event destruction");
   handleError(cudaEventDestroy(mStopEvent), "Stop event destruction");
}

double CudaTimer::start() {
   if (mEventPending == true) {
      WarnLog() << "CudaTimer called start() while event was still pending (timer message = \""
                << message << "\").\n";
   }
   handleError(cudaEventRecord(mStartEvent, mStream), "Recording start event");
   return 0;
}

double CudaTimer::stop() {
   handleError(cudaEventRecord(mStopEvent, mStream), "Recording stop event");
   if (mEventPending == true) {
      WarnLog() << "CudaTimer called stop() while event was still pending (timer message = \""
                << message << "\").\n";
   }
   mEventPending = true;
   return 0;
}

// Note this function is blocking
double CudaTimer::accumulateTime() {
   if (mEventPending) {
      float curTime;
      handleError(cudaEventSynchronize(mStopEvent), "Synchronizing stop event");
      handleError(
            cudaEventElapsedTime(&curTime, mStartEvent, mStopEvent), "Calculating elapsed time");
      // Roundoff errors?
      mTime += curTime;
      mEventPending = false;
   }
   return (double)mTime;
}

int CudaTimer::fprint_time(PrintStream &printStream) const {
   if (rank == 0) {
      printStream << message << "processor cycle time == " << mTime << std::endl;
   }
   return 0;
}

} // namespace PV
