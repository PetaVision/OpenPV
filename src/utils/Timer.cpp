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

#include <cstdlib>             // size_t
#include <cstring>             // std::strlen
#include <ostream>             // std::endl

#ifdef PV_TIMER_VERBOSE
#include <cinttypes>           // PRIu64 printf directive
#endif // PV_TIMER_VERBOSE

#ifdef __APPLE__
#define USE_MACH_TIMER
#endif

#ifdef USE_MACH_TIMER
//#  include <CoreServices/CoreServices.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#else
#include <time.h>               // clock_gettime(), struct timespec
#endif // USE_MACH_TIMER

namespace PV {

/**
 * Convert to milliseconds
 */
static uint64_t get_cpu_time() {
#ifdef USE_MACH_TIMER
   return mach_absolute_time();
#else
   struct timespec tim;
   clock_gettime(CLOCK_REALTIME,&tim);
   return (uint64_t)tim.tv_sec * (uint64_t)1000000 + (uint64_t)((tim.tv_nsec+500L)/1000L);
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
#ifdef PV_TIMER_VERBOSE
   if (mEpoch == (uint64_t)0) { // Fix this Y2K-like problem before 586438 AD.
      mEpoch = get_cpu_time();
   }
#endif // PV_TIMER_VERBOSE

   mRank = 0;
   reset(init_time);
   mMessage = timermessage ? timermessage : "";
}

Timer::Timer(const char *objname, const char *objtype, const char *timertype, double init_time) {
#ifdef PV_TIMER_VERBOSE
   if (mEpoch == (uint64_t)0) {
      mEpoch = get_cpu_time();
   }
#endif // PV_TIMER_VERBOSE
   mRank = 0;
   reset(init_time);
   // C-style code was
   // sprintf(message, "%32s: total time in %-10s %-10s : ", objname, objtype, timertype);
   // plus memory memory management. Instead, let the C++ string class manage the memory.
   mMessage.clear();
   mMessage.reserve(72);
   stringPad(mMessage, std::strlen(objname), 32UL);
   mMessage.append(objname);
   mMessage.append(": total time in ");
   mMessage.append(objtype);
   stringPad(mMessage, std::strlen(objtype), 10UL);
   mMessage.append(" ");
   mMessage.append(timertype);
   stringPad(mMessage, std::strlen(timertype), 10UL);
   mMessage.append(" : ");
}

void Timer::stringPad(std::string &str, std::size_t fillCount, std::size_t padCount, char c) {
   if (fillCount < padCount) {
      auto stringSize = str.size();
      str.resize(stringSize + padCount - fillCount, c);
   }
}

Timer::~Timer() {}

void Timer::reset(double init_time) {
   mTimeStart   = get_cpu_time();
   mTimeEnd     = mTimeStart;
   mTimeElapsed = init_time;
}

double Timer::start() {
   mTimeStart = get_cpu_time();
   mRunning   = true;
#ifdef PV_TIMER_VERBOSE
   InfoLog().printf("%12" PRIu64 " Start %s\n", mTimeStart - mEpoch, mMessage.c_str());
#endif // PV_TIMER_VERBOSE
   return (double)mTimeStart;
}

double Timer::stop() {
   mRunning = false;
   mTimeEnd = get_cpu_time();
   mTimeElapsed += mTimeEnd - mTimeStart;
#ifdef PV_TIMER_VERBOSE
   InfoLog().printf("%12" PRIu64 " Stop  %s\n", mTimeEnd - mEpoch, mMessage.c_str());
#endif // PV_TIMER_VERBOSE
   return (double)mTimeEnd;
}

double Timer::elapsed_time() const {
   return (double)(mTimeElapsed + (mRunning ? get_cpu_time()-mTimeStart : (uint64_t)0));
}

int Timer::fprint_time(PrintStream &stream) const {
   if (mRank == 0) {
      stream << mMessage.c_str() << "processor cycle time == "
             << (float)cpu_time_to_sec(elapsed_time()) << std::endl;
   }
   return 0;
}

#ifdef PV_TIMER_VERBOSE
   uint64_t Timer::mEpoch = (int64_t)0;
#endif // PV_TIMER_VERBOSE

} // namespace PV
