/*
 *  Timer.hpp
 *
 *  Created by Craig Rasmussen on 11/15/09.
 *  Copyright 2009 Los Alamos National Laboratory. All rights reserved.
 *
 */

#ifndef TIMER_HPP_
#define TIMER_HPP_

#include "io/PrintStream.hpp"

#include <cstdint>      // uint64_t
#include <cstdlib>      // size_t
#include <string>       // string class

////////////////////////////////////////////////////////////////////////////////

namespace PV {

uint64_t get_cpu_time();
static double cpu_time_to_sec(uint64_t cpu_elapsed);

class Timer {
  public:
   Timer(const char *timermessage, double init_time = 0.0);
   Timer(const char *objname, const char *objtype, const char *timertype, double init_time = 0.0);
   virtual ~Timer();
   void reset(double init_time = 0.0);

   virtual double start();
   virtual double stop();
   inline double elapsed_time() const;
   virtual int fprint_time(PrintStream &stream) const;

   static void stringPad(
         std::string &str, std::size_t fillCount, std::size_t padCount, char c = ' ');

  protected:
   int mRank;
   std::string mMessage;

   bool mRunning = false; // start() sets running flag to true; stop() sets it to false.
   uint64_t mTimeStart, mTimeEnd;
   uint64_t mTimeElapsed = (uint64_t)0;

#ifdef PV_TIMER_VERBOSE
   static uint64_t mEpoch;
#endif // PV_TIMER_VERBOSE
};

} // namespace PV

#endif /* TIMER_HPP_ */
