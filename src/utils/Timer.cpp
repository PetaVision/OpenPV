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
#define USE_MACH_TIMER
#endif

#ifdef USE_MACH_TIMER
//#  include <CoreServices/CoreServices.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#else
#include <time.h>
#endif // USE_MACH_TIMER

namespace PV {

/**
 * Convert to milliseconds
 */
uint64_t get_cpu_time() {
#ifdef USE_MACH_TIMER
   return mach_absolute_time();
#else
   struct timespec tim;
   clock_gettime(CLOCK_REALTIME,&tim);
   return ((uint64_t)tim.tv_sec) * 1000000 + (uint64_t)((tim.tv_nsec+500L)/1000L);
#endif
}

// Convert to milliseconds
static double cpu_time_to_sec(uint64_t cpu_elapsed) {
   double us = 0.0;
#ifdef USE_MACH_TIMER
   static mach_timebase_info_data_t info;
   mach_timebase_info(&info);
   cpu_elapsed *= info.numer;
   cpu_elapsed /= info.denom;
   us = (double)(cpu_elapsed / 1000); // microseconds
#else
   us = (double)cpu_elapsed;
#endif
   return us / 1000.0;
}

Timer::Timer(const char *timermessage, double init_time) {
   rank = 0;
   reset(init_time);
   message = strdup(timermessage ? timermessage : "");
}

Timer::Timer(const char *objname, const char *objtype, const char *timertype, double init_time) {
   rank = 0;
   reset(init_time);
   int charsneeded =
         snprintf(nullptr, 0, "%32s: total time in %6s %10s: ", objname, objtype, timertype) + 1;
   message = (char *)malloc(charsneeded);
   FatalIf(
         message == nullptr,
         "Timer::setMessage unable to allocate memory for Timer message: called with name=%s, "
         "objtype=%s, timertype=%s\n",
         objname,
         objtype,
         timertype);
#ifdef NDEBUG
   snprintf(message, charsneeded, "%32s: total time in %6s %10s: ", objname, objtype, timertype);
#else
   int chars_used =
         snprintf(
               message, charsneeded, "%32s: total time in %6s %10s: ", objname, objtype, timertype)
         + 1;
   assert(chars_used <= charsneeded);
#endif // NDEBUG
}

Timer::~Timer() { free(message); }

void Timer::reset(double init_time) {
   time_start   = get_cpu_time();
   time_end     = time_start;
   time_elapsed = init_time;
}

double Timer::start() {
   time_start = get_cpu_time();
   running    = true;
   return (double)time_start;
}

double Timer::stop() {
   running = false;
   time_end = get_cpu_time();
   time_elapsed += time_end - time_start;
   return (double)time_end;
}

double Timer::elapsed_time() const {
   return (double)(time_elapsed + (running ? get_cpu_time()-time_start : (uint64_t)0));
}

int Timer::fprint_time(PrintStream &stream) const {
   if (rank == 0) {
      stream << message << "processor cycle time == " << (float)cpu_time_to_sec(elapsed_time())
             << std::endl;
   }
   return 0;
}

} // namespace PV
