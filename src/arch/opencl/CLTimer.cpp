/*
 *  CLTimer.cu
 *
 *  Sheng Lundquist
 */

#include "CLTimer.hpp"

namespace PV{

#ifdef PV_USE_OPENCL

CLTimer::CLTimer(double init_time):Timer(init_time)
{
   time = 0;
   timerEvent = new cl_event();
}

CLTimer::CLTimer(const char * timermessage, double init_time):Timer(timermessage, init_time)
{
   time = 0;
   timerEvent = new cl_event();
}

CLTimer::CLTimer(const char * objname, const char * objtype, const char * timertype, double init_time):Timer(objname, objtype, timertype, init_time){
   time = 0;
   timerEvent = new cl_event();
}

CLTimer::~CLTimer()
{
   if(timerEvent){
      clReleaseEvent(*timerEvent);
   }
}

//Note this function is blocking
double CLTimer::accumulateTime(){
   if(timerEvent){
      cl_ulong time_start, time_end;
      clWaitForEvents(1, timerEvent);
      clGetEventProfilingInfo(*timerEvent, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
      clGetEventProfilingInfo(*timerEvent, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
      //Roundoff errors?
      time += (time_end - time_start)/1000000;
   }
   return (double) time;
}

int CLTimer::fprint_time(FILE * stream) {
   if (rank == 0) {
      fprintf(stream, "%sprocessor cycle time == %f\n", message, time);
      fflush(stream);
   }
   return 0;
}
#endif
 
}  // namespace PV
