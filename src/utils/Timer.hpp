/*
 *  Timer.hpp
 *
 *  Created by Craig Rasmussen on 11/15/09.
 *  Copyright 2009 Los Alamos National Laboratory. All rights reserved.
 *
 */

#ifndef TIMER_HPP_
#define TIMER_HPP_

#define MACH_TIMER

#ifdef MACH_TIMER
#include <CoreServices/CoreServices.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#endif // MACH_TIMER


////////////////////////////////////////////////////////////////////////////////

namespace PV {
   
   class Timer {
   public:
      Timer();
      virtual ~Timer();
      
      virtual double start();
      virtual double stop();
      virtual double elapsed_time();
      
   protected:
      
      int rank;
      
      uint64_t     mach_start, mach_end;
      double       mach_elapsed;

   };
   
} // namespace PV

#endif /* TIMER_HPP_ */
