/*
 *  Timer.cpp
 *  opencl
 *
 *  Created by Craig Rasmussen on 11/15/09.
 *  Copyright 2009 Los Alamos National Laboratory. All rights reserved.
 *
 */

#include "Timer.hpp"

double mach_time_to_sec(uint64_t elapsed);

namespace PV {

Timer::Timer()
{
   rank = 0;
   mach_start   = mach_absolute_time();
   mach_end     = mach_start;
   mach_elapsed = 0.0;
}

Timer::~Timer()
{
}

double Timer::start()
{
   return (mach_start = mach_absolute_time());
}

double Timer::stop()
{
   mach_end = mach_absolute_time();
   mach_elapsed += mach_end - mach_start;
   return mach_end;
}

double Timer::elapsed_time()
{
   if (rank == 0) {
      fprintf(stdout, "Mach processor cycle time == %f\n", (float) mach_time_to_sec(mach_elapsed));
      fflush(stdout);
   }
   return mach_elapsed;
}
   
}  // namespace PV


// Convert to milliseconds
double mach_time_to_sec(uint64_t elapsed)
{
   double ms;
   static mach_timebase_info_data_t  sTimebaseInfo;
   
   if ( sTimebaseInfo.denom == 0 ) {
      // initialize (yuk, hope it isn't some stray value)
      (void) mach_timebase_info(&sTimebaseInfo);
   }
   
   ms = (double) (elapsed) / 1.0e9;
   ms *= sTimebaseInfo.numer / sTimebaseInfo.denom;
   
   return ms;
}
