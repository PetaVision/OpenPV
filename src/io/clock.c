#include <sys/time.h>
#include <sys/times.h>
#include <sys/resource.h>
#include <time.h>
#include <stdio.h>
#include <inttypes.h>

#define MACH_TIMER 1
#undef  MACH_TIMER

#ifdef MACH_TIMER
#include <CoreServices/CoreServices.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#endif // MACH_TIMER

/**
 *  compile with /opt/cell/bin/ppu-gcc -m32 -c clock.c
 */

/* global variables */
static clock_t start, end;
static double rstart, tstart, rend, tend;
static struct timeval tv_start, tv_end;
#ifdef CYCLE_TIMER
static uint64_t cycle_start, cycle_end;
#endif
#ifdef MACH_TIMER
static uint64_t     mach_start, mach_end;
static double       mach_elapsed;
#endif

#define OPTERON 1
#undef  OPTERON

#ifdef OPTERON

#define CYCLES_PER_SEC 2004591000

#  define READ_CYCLE_COUNTER(VAR)                     \
  do {                                                \
    uint32_t lo, hi;                                  \
    asm volatile("rdtsc" : "=a" (lo), "=d" (hi));     \
    VAR = ((uint64_t)hi <<32) | lo;                   \
  } while(0)

#endif

#ifdef PPU

// This variable is timebase in /procs/cpu_info on cell
//#define CYCLES_PER_SEC 14318000   // from cellbuzz at Georgia Tech?
#define CYCLES_PER_SEC 26666666

#  define READ_CYCLE_COUNTER(VAR)                             \
  do {                                                        \
    uint32_t tbu1, tbu2, tbl;                                 \
    asm volatile ("\n"                                        \
                    "0:\n"                                    \
                    "\tmftbu\t%0\n"                           \
                    "\tmftb\t%2\n"                            \
                    "\tmftbu\t%1\n"                           \
                    "\tcmpw\t%1,%0\n"                         \
                    "\tbne\t0b"                               \
		  : "=r" (tbu1), "=r" (tbu2), "=r" (tbl));    \
    VAR = ((uint64_t)tbu1 << 32) | tbl;                       \
  } while (0)

#endif

#ifdef MACH_TIMER
double mach_time_to_sec(uint64_t elapsed);
#endif

void start_clock()
{
  struct rusage r;
  struct timeval t;

  start = clock();

  getrusage(RUSAGE_SELF, &r);
  rstart = r.ru_utime.tv_sec + r.ru_utime.tv_usec*1.0e-6;

  gettimeofday( &t, (struct timezone *)0 );
  tstart = t.tv_sec + t.tv_usec*1.0e-6;
  tv_start = t;

#ifdef MACH_TIMER
    mach_start = mach_absolute_time();
#endif

#ifdef CYCLE_TIMER
  READ_CYCLE_COUNTER(cycle_start);
#endif
}

void stop_clock()
{
  struct rusage r;
  struct timeval t;

  end = clock();

  getrusage(RUSAGE_SELF, &r);
  rend = r.ru_utime.tv_sec + r.ru_utime.tv_usec*1.0e-6;

  gettimeofday( &t, (struct timezone *)0 );
  tend = t.tv_sec + t.tv_usec*1.0e-6;
  tv_end = t;

#ifdef MACH_TIMER
    mach_end = mach_absolute_time();
#endif

#ifdef CYCLE_TIMER
  READ_CYCLE_COUNTER(cycle_end);
#endif

  fprintf(stdout, "     %1.2f seconds elapsed (clock())\n",
	  (float)((double)(end-start) / CLOCKS_PER_SEC));
  fprintf(stdout, "     %1.2f seconds elapsed (CPU)\n", (float)(rend-rstart));
  fprintf(stdout, "     %1.2f seconds elapsed (wallclock)\n", (float)(tend-tstart));
  fflush(stdout);

#ifdef MACH_TIMER
    mach_elapsed = mach_time_to_sec(mach_end - mach_start);
    fprintf(stderr, "Mach processor cycle time = %f\n", (float) mach_elapsed);
#endif

#ifdef CYCLE_TIMER
  fprintf(stderr, "%ld cycles elapsed\n", (long) (cycle_end - cycle_start));
  fprintf(stderr, "%f seconds elapsed (cycle timer)\n",
	  (float) ((double)(cycle_end - cycle_start))/CYCLES_PER_SEC);
#endif
}

double elapsed_time()
{
   double elapsed = 0.0;
#ifdef MACH_TIMER
   elapsed = mach_time_to_sec(mach_absolute_time() - mach_start);
#endif
   return elapsed;
}


#ifdef MACH_TIMER
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

#endif
