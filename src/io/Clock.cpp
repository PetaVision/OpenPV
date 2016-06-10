/*
 * Clock.cpp
 *
 *  Created on: Jun 6, 2016
 *      Author: pschultz
 */

#include <io/Clock.hpp>

namespace PV {

void Clock::start_clock() {
   struct rusage r;
   struct timeval t;

   m_start = clock();

   getrusage(RUSAGE_SELF, &r);
   m_rstart = r.ru_utime.tv_sec + r.ru_utime.tv_usec*1.0e-6;

   gettimeofday( &t, (struct timezone *)0 );
   m_tstart = t.tv_sec + t.tv_usec*1.0e-6;
   m_tv_start = t;
#ifdef MACH_TIMER
   m_mach_start = mach_absolute_time();
#endif

#ifdef CYCLE_TIMER
  READ_CYCLE_COUNTER(cycle_start);
#endif
}

void Clock::stop_clock() {
   struct rusage r;
   struct timeval t;

   m_end = clock();

   getrusage(RUSAGE_SELF, &r);
   m_rend = r.ru_utime.tv_sec + r.ru_utime.tv_usec*1.0e-6;

   gettimeofday( &t, (struct timezone *)0 );
   m_tend = t.tv_sec + t.tv_usec*1.0e-6;
   m_tv_end = t;

#ifdef MACH_TIMER
   m_mach_end = mach_absolute_time();
#endif

#ifdef CYCLE_TIMER
   READ_CYCLE_COUNTER(m_cycle_end);
#endif
}

void Clock::print_elapsed(std::ostream& stream) {
   std::streamsize w = stream.width(1);
   std::streamsize p = stream.precision(2);
   stream << "     " << (float)((double)(m_end-m_start) / CLOCKS_PER_SEC) << " seconds elapsed (clock())\n";
   stream << "     " << (float)(m_rend-m_rstart) << " seconds elapsed (CPU)\n";
   stream << "     " << (float)(m_tend-m_tstart) << " seconds elapsed (wallclock)\n";
   stream.width(w);
   stream.precision(p);

#ifdef MACH_TIMER
    uint64_t mach_elapsed = mach_time_to_sec(m_mach_end - m_mach_start);
    stream << "Mach processor cycle time = " << (float) m_mach_elapsed << "\n";
#endif

#ifdef CYCLE_TIMER
   uint64_t cycle_elapsed = m_cycle_end - m_cycle_start;
   stream << cycle_elapsed << " cycles elapsed\n";
   stream << (float) ((double)cycle_elapsed/(double)CYCLES_PER_SEC << " seconds elapsed (cycle timer)\n";
#endif
}

#ifdef MACH_TIMER
double Clock::elapsed_time() {
   return mach_time_to_sec(mach_absolute_time() - mach_start);
}

double Clock::mach_time_to_sec(uint64_t elapsed)
{
  if ( sTimebaseInfo.denom == 0 ) {
    // initialize (yuk, hope it isn't some stray value)
    (void) mach_timebase_info(&sTimebaseInfo);
  }

  double ms = (double) (elapsed) / 1.0e9;
  ms *= sTimebaseInfo.numer / sTimebaseInfo.denom;

  return ms;
}

#endif // MACH_TIMER

}  // namespace PV


