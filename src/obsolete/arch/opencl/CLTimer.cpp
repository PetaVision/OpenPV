/*
 *  CLTimer.cu
 *
 *  Sheng Lundquist
 */

#include "CLTimer.hpp"
#include "utils/PVLog.hpp"
#include "utils/PVAssert.hpp"

namespace PV{

#ifdef PV_USE_OPENCL

CLTimer::CLTimer(cl_command_queue commands, double init_time):Timer(init_time)
{
   time = 0;
   this->commands = commands;
   startEvent = new cl_event();
   stopEvent = new cl_event();
}

CLTimer::CLTimer(cl_command_queue commands, const char * timermessage, double init_time):Timer(timermessage, init_time)
{
   time = 0;
   this->commands = commands;
   startEvent = new cl_event();
   stopEvent = new cl_event();
}

CLTimer::CLTimer(cl_command_queue commands, const char * objname, const char * objtype, const char * timertype, double init_time):Timer(objname, objtype, timertype, init_time){
   time = 0;
   this->commands = commands;
   startEvent = new cl_event();
   stopEvent = new cl_event();
}

CLTimer::~CLTimer()
{
   if(startEvent){
      clReleaseEvent(*startEvent);
   }
   if(stopEvent){
      clReleaseEvent(*stopEvent);
   }
   delete(startEvent);
   delete(stopEvent);
}


//Note this function is blocking
double CLTimer::accumulateTime(){
   //pvInfo().printf("Accumulating opencl timer\n");
   cl_ulong cl_time_start, cl_time_end;
   cl_time_start = 0;
   cl_time_end = 0;
   //clWaitForEvents(1, &stopEvent);
   clFinish(commands);
   cl_int status = clGetEventProfilingInfo(*startEvent, CL_PROFILING_COMMAND_START, sizeof(cl_time_start), &cl_time_start, NULL);
   if(status != CL_SUCCESS){
      pvError(errorMessage);
      switch(status){
         case CL_PROFILING_INFO_NOT_AVAILABLE:
            errorMessage.printf("Profiling info not avaliable\n");
            break;
         case CL_INVALID_VALUE:
            errorMessage.printf("Invalid param names\n");
            break;
         case CL_INVALID_EVENT:
            errorMessage.printf("Invalid event for start event\n");
            break;
         default:
            errorMessage.printf("Unknown error\n");
      }
      exit(EXIT_FAILURE);
   }
   status = clGetEventProfilingInfo(*stopEvent, CL_PROFILING_COMMAND_END, sizeof(cl_time_end), &cl_time_end, NULL);
   if(status == CL_INVALID_EVENT){
      status = clGetEventProfilingInfo(*startEvent, CL_PROFILING_COMMAND_END, sizeof(cl_time_end), &cl_time_end, NULL);
   }
   if(status != CL_SUCCESS){
      pvError(errorMessage);
      switch(status){
         case CL_PROFILING_INFO_NOT_AVAILABLE:
            errorMessage.printf("Profiling info not avaliable\n");
            break;
         case CL_INVALID_VALUE:
            errorMessage.printf("Invalid param names\n");
            break;
         case CL_INVALID_EVENT:
            errorMessage.printf("Invalid event for stop event\n");
            break;
         default:
            errorMessage.printf("Unknown error\n");
      }
      exit(EXIT_FAILURE);
   }
   //Roundoff errors?
   //pvInfo().printf("Diff times: %f\n", (double)(cl_time_end - cl_time_start)/1000000);
   time += (double)(cl_time_end - cl_time_start)/1000000;

   return (double) time;
}

int CLTimer::fprint_time(std::ostream& stream) {
   if (rank == 0) {
      stream << message << "processor cycle time == " << time << std::endl;
   }
   return 0;
}
#endif
 
}  // namespace PV
