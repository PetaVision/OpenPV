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
}

CLTimer::CLTimer(const char * timermessage, double init_time):Timer(timermessage, init_time)
{
   time = 0;
}

CLTimer::CLTimer(const char * objname, const char * objtype, const char * timertype, double init_time):Timer(objname, objtype, timertype, init_time){
   time = 0;
}

CLTimer::~CLTimer()
{
   if(startEvent){
      clReleaseEvent(startEvent);
   }
   if(stopEvent){
      clReleaseEvent(stopEvent);
   }
}

double CLTimer::start()
{
   cl_int status = clEnqueueMarker(commands, &startEvent);
   assert(status == CL_SUCCESS);
   return 0;
}

double CLTimer::stop()
{
   cl_int status = clEnqueueMarker(commands, &stopEvent);
   assert(status == CL_SUCCESS);
   return 0;
}

//Note this function is blocking
double CLTimer::accumulateTime(){
   if(startEvent && stopEvent){
      cl_ulong cl_time_start, cl_time_end;
      cl_time_start = 0;
      cl_time_end = 0;
      clWaitForEvents(1, &stopEvent);
      cl_int status = clGetEventProfilingInfo(startEvent, CL_PROFILING_COMMAND_START, sizeof(cl_time_start), &cl_time_start, NULL);
      assert(status == CL_SUCCESS);
      status = clGetEventProfilingInfo(stopEvent, CL_PROFILING_COMMAND_END, sizeof(cl_time_end), &cl_time_end, NULL);
      assert(status == CL_SUCCESS);
      //Roundoff errors?
      time += (double)(cl_time_end - cl_time_start)/1000000;
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
