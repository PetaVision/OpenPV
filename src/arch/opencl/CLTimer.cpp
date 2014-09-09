/*
 *  CLTimer.cu
 *
 *  Sheng Lundquist
 */

#include "CLTimer.hpp"

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

//void CLTimer::clearStopEvent(){
//   clReleaseEvent(*stopEvent);
//}

//double CLTimer::start()
//{
//   printf("Starting opencl timer\n");
//   cl_int status = clEnqueueMarkerWithWaitList(commands, 0, NULL, startEvent);
//   if(status != CL_SUCCESS){
//      switch(status){
//         case CL_INVALID_COMMAND_QUEUE:
//            printf("Invalid command queue\n");
//            break;
//         case CL_INVALID_EVENT_WAIT_LIST:
//            printf("Invalid event wait list\n");
//            break;
//         case CL_OUT_OF_RESOURCES:
//            printf("Out of resources\n");
//            break;
//         case CL_OUT_OF_HOST_MEMORY:
//            printf("Out of host memory\n");
//            break;
//         default:
//            printf("Unknown error\n");
//      }
//      exit(EXIT_FAILURE);
//   }
//   return 0;
//}
//
//double CLTimer::stop()
//{
//   printf("Stopping opencl timer\n");
//   cl_int status = clEnqueueMarkerWithWaitList(commands, 0, NULL, stopEvent);
//   if(status != CL_SUCCESS){
//      switch(status){
//         case CL_INVALID_COMMAND_QUEUE:
//            printf("Invalid command queue\n");
//            break;
//         case CL_INVALID_EVENT_WAIT_LIST:
//            printf("Invalid event wait list\n");
//            break;
//         case CL_OUT_OF_RESOURCES:
//            printf("Out of resources\n");
//            break;
//         case CL_OUT_OF_HOST_MEMORY:
//            printf("Out of host memory\n");
//            break;
//         default:
//            printf("Unknown error\n");
//      }
//      exit(EXIT_FAILURE);
//   }
//   return 0;
//}

//Note this function is blocking
double CLTimer::accumulateTime(){
   //printf("Accumulating opencl timer\n");
   cl_ulong cl_time_start, cl_time_end;
   cl_time_start = 0;
   cl_time_end = 0;
   //clWaitForEvents(1, &stopEvent);
   clFinish(commands);
   cl_int status = clGetEventProfilingInfo(*startEvent, CL_PROFILING_COMMAND_START, sizeof(cl_time_start), &cl_time_start, NULL);
   if(status != CL_SUCCESS){
      switch(status){
         case CL_PROFILING_INFO_NOT_AVAILABLE:
            printf("Profiling info not avaliable\n");
            break;
         case CL_INVALID_VALUE:
            printf("Invalid param names\n");
            break;
         case CL_INVALID_EVENT:
            printf("Invalid event for start event\n");
            break;
         default:
            printf("Unknown error\n");
      }
      exit(EXIT_FAILURE);
   }
   status = clGetEventProfilingInfo(*stopEvent, CL_PROFILING_COMMAND_END, sizeof(cl_time_end), &cl_time_end, NULL);
   if(status == CL_INVALID_EVENT){
      status = clGetEventProfilingInfo(*startEvent, CL_PROFILING_COMMAND_END, sizeof(cl_time_end), &cl_time_end, NULL);
   }
   if(status != CL_SUCCESS){
      switch(status){
         case CL_PROFILING_INFO_NOT_AVAILABLE:
            printf("Profiling info not avaliable\n");
            break;
         case CL_INVALID_VALUE:
            printf("Invalid param names\n");
            break;
         case CL_INVALID_EVENT:
            printf("Invalid event for stop event\n");
            break;
         default:
            printf("Unknown error\n");
      }
      exit(EXIT_FAILURE);
   }
   //Roundoff errors?
   //printf("Diff times: %f\n", (double)(cl_time_end - cl_time_start)/1000000);
   time += (double)(cl_time_end - cl_time_start)/1000000;

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
