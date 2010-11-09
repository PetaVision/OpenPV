/*
 *  Timer.cpp
 *  opencl
 *
 *  Created by Craig Rasmussen on 11/15/09.
 *  Copyright 2009 Los Alamos National Laboratory. All rights reserved.
 *
 */

#include "Timer.hpp"
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
#endif USE_MACH_TIMER


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
   //printf("get_cpu_time: sec==%d usec==%d\n", tim.tv_sec, tim.tv_usec);
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

Timer::Timer()
{
   rank = 0;
   time_start   = get_cpu_time();
   time_end     = time_start;
   time_elapsed = 0.0;
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
   if (rank == 0) {
      fprintf(stdout, "processor cycle time == %f\n", (float) cpu_time_to_sec(time_elapsed));
      fflush(stdout);
   }
   return (double) time_elapsed;
}
   
}  // namespace PV
