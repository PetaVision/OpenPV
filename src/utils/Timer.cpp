/*
 *  Timer.cpp
 *  opencl
 *
 *  Created by Craig Rasmussen on 11/15/09.
 *  Copyright 2009 Los Alamos National Laboratory. All rights reserved.
 *
 */

#include "Timer.hpp"
#include "utils/PVLog.hpp"
#include <stdio.h>

#ifdef __APPLE__
#  define USE_MACH_TIMER
#endif

#ifdef USE_MACH_TIMER
//#  include <CoreServices/CoreServices.h>
#  include <mach/mach.h>
#  include <mach/mach_time.h>
#else
#  include <sys/time.h>
#endif // USE_MACH_TIMER


/**
 * Convert to milliseconds
 */
uint64_t get_cpu_time() {
#ifdef USE_MACH_TIMER
   return mach_absolute_time();
#else
   struct timeval tim;
   //   struct rusage ru;
   //   getrusage(RUSAGE_SELF, &ru);
   //   tim = ru.ru_utime;
   gettimeofday(&tim, NULL);
   //fprintf(stdout, "get_cpu_time: sec==%d usec==%d\n", tim.tv_sec, tim.tv_usec);
   return ((uint64_t) tim.tv_sec)*1000000 + (uint64_t) tim.tv_usec;
#endif
}

// Convert to milliseconds
static double cpu_time_to_sec(uint64_t cpu_elapsed)
{
   double us = 0.0;
#ifdef USE_MACH_TIMER
   static mach_timebase_info_data_t  info;
   mach_timebase_info(&info);
   cpu_elapsed *= info.numer;
   cpu_elapsed /= info.denom;
   us = (double) (cpu_elapsed/1000);  // microseconds
#else
   us = (double) cpu_elapsed;
#endif
   return us/1000.0;
}


namespace PV {

Timer::Timer(double init_time)
{
   rank = 0;
   reset(init_time);
   message = strdup("");
}

Timer::Timer(const char * timermessage, double init_time)
{
   rank = 0;
   reset(init_time);
   message = strdup(timermessage ? timermessage : "");
}

Timer::Timer(const char * objname, const char * objtype, const char * timertype, double init_time) {
   rank = 0;
   reset(init_time);
   char dummy;
   int charsneeded = snprintf(&dummy, 1, "%32s: total time in %6s %10s: ", objname, objtype, timertype);
   message = (char *) malloc(charsneeded+1);
   if (message==NULL) {
      pvError().printf("Timer::setMessage unable to allocate memory for Timer message: called with name=%s, objtype=%s, timertype=%s\n", objname, objtype, timertype);
   }
   int chars_used = snprintf(message, charsneeded+1, "%32s: total time in %6s %10s: ", objname, objtype, timertype);
   assert(chars_used<=charsneeded);
}

Timer::~Timer()
{
   free(message);
}

void Timer::reset(double init_time)
{
   time_start   = get_cpu_time();
   time_end     = time_start;
   time_elapsed = init_time;
}

double Timer::start()
{
   return (double) (time_start = get_cpu_time());
}

double Timer::stop()
{
   time_end = get_cpu_time();
   time_elapsed += time_end - time_start;
   return (double) time_end;
}

double Timer::elapsed_time()
{
   return (double) time_elapsed;
}

int Timer::fprint_time(std::ostream& stream) {
   if (rank==0) {
      stream << message << "processor cycle time == " << (float) cpu_time_to_sec(elapsed_time()) << std::endl;
   }
   return 0;
}
 
}  // namespace PV
