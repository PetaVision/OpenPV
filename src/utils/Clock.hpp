/*
 * Clock.hpp
 *
 *  Created on: Jun 6, 2016
 *      Author: pschultz
 */

#ifndef CLOCK_HPP_
#define CLOCK_HPP_

#include <ostream>
#include <stdint.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <time.h>

// #define MACH_TIMER 1
#undef MACH_TIMER
#ifdef MACH_TIMER
#include <CoreServices/CoreServices.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#endif // MACH_TIMER

// #define CYCLE_TIMER 1
#undef CYCLE_TIMER

namespace PV {

class Clock {
  public:
   void start_clock();
   void stop_clock();
   void print_elapsed(std::ostream &stream);
#ifdef MACH_TIMER
   double elapsed_time();
   double mach_time_to_sec(uint64_t elapsed);
#endif // MACH_TIMER

   // Data members
  private:
   clock_t m_start;
   clock_t m_end;
   double m_rstart;
   double m_rend;
   double m_tstart;
   double m_tend;

#ifdef CYCLE_TIMER
   uint64_t m_cycle_start;
   uint64_t m_cycle_end;
#endif

#ifdef MACH_TIMER
   uint64_t m_mach_start;
   uint64_t m_mach_end;
   mach_timebase_info_data_t sTimebaseInfo;
#endif
};

} // namespace PV

#endif /* CLOCK_HPP_ */
