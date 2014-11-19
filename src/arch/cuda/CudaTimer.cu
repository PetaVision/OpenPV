/*
 *  CudaTimer.cu
 *
 *  Sheng Lundquist
 */

#include "CudaTimer.hpp"
#include "cuda_util.hpp"


namespace PVCuda {

CudaTimer::CudaTimer(double init_time):PV::Timer(init_time)
{
   handleError(cudaEventCreate(&startEvent), "Start event creation");
   handleError(cudaEventCreate(&stopEvent), "Stop event creation");
   time = 0;
}

CudaTimer::CudaTimer(const char * timermessage, double init_time):PV::Timer(timermessage, init_time)
{
   handleError(cudaEventCreate(&startEvent), "Start even creation");
   handleError(cudaEventCreate(&stopEvent), "Stop event creation");
   time = 0;
}

CudaTimer::CudaTimer(const char * objname, const char * objtype, const char * timertype, double init_time):PV::Timer(objname, objtype, timertype, init_time){
   handleError(cudaEventCreate(&startEvent), "Start event creation");
   handleError(cudaEventCreate(&stopEvent), "Stop event creation");
   time = 0;
}

CudaTimer::~CudaTimer()
{
   handleError(cudaEventDestroy(startEvent), "Start event destruction");
   handleError(cudaEventDestroy(stopEvent), "Stop event destruction");
}

double CudaTimer::start()
{
   handleError(cudaEventRecord(startEvent, stream), "Recording start event");
   return 0;
}

double CudaTimer::stop()
{
   handleError(cudaEventRecord(stopEvent, stream), "Recording stop event");
   return 0;
}

//Note this function is blocking
double CudaTimer::accumulateTime(){
   float curTime;
   handleError(cudaEventSynchronize(stopEvent), "Synchronizing stop event");
   handleError(cudaEventElapsedTime(&curTime, startEvent, stopEvent), "Calculating elapsed time");
   //Roundoff errors?
   time += curTime;
   return (double) time;
}

int CudaTimer::fprint_time(FILE * stream) {
   if (rank == 0) {
      fprintf(stream, "%sprocessor cycle time == %f\n", message, time);
      fflush(stream);
   }
   return 0;
}
 
}  // namespace PV
